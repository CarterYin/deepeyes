import os
import json
import math
import base64
import argparse
import shutil
import multiprocessing
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
import requests

multiprocessing.set_start_method("spawn", force=True)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="qwen", help="Model name for result save")
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument("--api_url", type=str, default="http://10.39.19.140:8000/v1", help="API URL")
parser.add_argument("--vstar_bench_path", type=str, default=None, help="Path to the V* benchmark")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the results")
parser.add_argument("--result_tag", type=str, default="zoom_aug", help="Suffix tag for output result files")
parser.add_argument("--zoom_save_path", type=str, default=None, help="Path to save zoomed crop images")
parser.add_argument("--save_zoom_images", type=str2bool, default=True, help="Whether to save zoomed crop images")
parser.add_argument("--build_zoom_dataset", type=str2bool, default=True, help="Whether to build a new dataset with zoom images")
parser.add_argument("--new_dataset_suffix", type=str, default="with_zoom_aug", help="Suffix used for rebuilt dataset folder")
parser.add_argument("--zoom_output_width", type=int, default=448, help="Width of zoom images saved to new dataset")
parser.add_argument("--zoom_output_height", type=int, default=448, help="Height of zoom images saved to new dataset")
parser.add_argument("--zoom_duplicate_times", type=int, default=1, help="How many copies to add per zoom image into new dataset")
parser.add_argument("--eval_model_name", type=str, default=None, help="Model name for evaluation")
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()

if args.zoom_output_width <= 0 or args.zoom_output_height <= 0:
    raise ValueError("zoom_output_width/zoom_output_height must be > 0")
if args.zoom_duplicate_times <= 0:
    raise ValueError("zoom_duplicate_times must be > 0")

openai_api_key = args.api_key
openai_api_base = args.api_url
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models["data"][0]["id"]
else:
    eval_model_name = args.eval_model_name

vstar_bench_path = args.vstar_bench_path
save_path = os.path.join(args.save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)

if args.save_zoom_images:
    zoom_save_root = args.zoom_save_path if args.zoom_save_path else os.path.join(save_path, "zoom_crops")
    os.makedirs(zoom_save_root, exist_ok=True)
else:
    zoom_save_root = None

if args.build_zoom_dataset:
    vstar_abs = os.path.abspath(vstar_bench_path.rstrip(os.sep))
    rebuilt_dataset_root = os.path.join(os.path.dirname(vstar_abs), f"{os.path.basename(vstar_abs)}_{args.new_dataset_suffix}")
    os.makedirs(rebuilt_dataset_root, exist_ok=True)
else:
    rebuilt_dataset_root = None

abc_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

instruction_prompt_system = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""
USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
instruction_prompt_before = """Question: {question}
Options: {options}
""" + USER_PROMPT_V2

start_token = "<tool_call>"
end_token = "</tool_call>"


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS):
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _bbox_token(bbox):
    return "_".join(str(int(round(v))) for v in bbox)


def save_zoom_to_dataset(rebuilt_split_dir, cropped_image, img_stem, turn_idx, bbox, anno_path, duplicate_times):
    resized_zoom = cropped_image.resize((args.zoom_output_width, args.zoom_output_height), resample=Image.BICUBIC)
    saved_paths = []
    base_name = f"{img_stem}_turn{turn_idx:02d}_bbox_{_bbox_token(bbox)}"
    for dup_idx in range(duplicate_times):
        if duplicate_times == 1:
            image_name = f"{base_name}.png"
            anno_name = f"{base_name}.json"
        else:
            image_name = f"{base_name}_dup{dup_idx + 1:02d}.png"
            anno_name = f"{base_name}_dup{dup_idx + 1:02d}.json"
        image_path = os.path.join(rebuilt_split_dir, image_name)
        anno_dst = os.path.join(rebuilt_split_dir, anno_name)
        resized_zoom.save(image_path)
        shutil.copy2(anno_path, anno_dst)
        saved_paths.append(image_path)
    return saved_paths


def process(img_arg):
    img, test_path = img_arg
    img_path = os.path.join(test_path, img)
    img_stem = os.path.splitext(img)[0]
    anno_path = os.path.join(test_path, f"{img_stem}.json")
    if not os.path.exists(anno_path):
        return None

    with open(anno_path, "r") as f:
        anno = json.load(f)
    question = anno["question"]
    options = anno["options"]

    option_str = "\n"
    for i in range(len(options)):
        option_str += abc_map[i + 1] + ". " + options[i] + "\n"

    prompt = instruction_prompt_before.format(question=question, options=option_str)
    pil_img = Image.open(img_path)
    base64_image = encode_image_to_base64(img_path)

    chat_message = [
        {"role": "system", "content": instruction_prompt_system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    print_messages = [
        {"role": "system", "content": instruction_prompt_system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    response_message = ""
    zoom_image_paths = []
    rebuilt_dataset_zoom_image_paths = []
    rebuilt_dataset_image_path = None

    test_split = os.path.basename(test_path)
    rebuilt_split_dir = None
    if args.build_zoom_dataset:
        rebuilt_split_dir = os.path.join(rebuilt_dataset_root, test_split)
        os.makedirs(rebuilt_split_dir, exist_ok=True)
        rebuilt_img_path = os.path.join(rebuilt_split_dir, img)
        rebuilt_anno_path = os.path.join(rebuilt_split_dir, os.path.basename(anno_path))
        if not os.path.exists(rebuilt_img_path):
            shutil.copy2(img_path, rebuilt_img_path)
        if not os.path.exists(rebuilt_anno_path):
            shutil.copy2(anno_path, rebuilt_anno_path)
        rebuilt_dataset_image_path = rebuilt_img_path

    status = "success"
    try_count = 0
    turn_idx = 0
    try:
        while "</answer>" not in response_message:
            if "</answer>" in response_message and "<answer>" in response_message:
                break
            if try_count > 10:
                break

            response = client.chat.completions.create(
                model=eval_model_name,
                messages=chat_message,
                temperature=0.0,
                max_tokens=1024,
                stop=["<|im_end|>\n".strip()],
            )
            response_message = response.choices[0].message.content

            if start_token in response_message and end_token in response_message:
                action_list = eval(response_message.split(start_token)[1].split(end_token)[0].strip())
                bbox = action_list["arguments"]["bbox_2d"]
                left, top, right, bottom = bbox
                cropped_image = pil_img.crop((left, top, right, bottom))
                new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

                zoom_filename = f"{img_stem}_turn{turn_idx:02d}_bbox_{_bbox_token(bbox)}.png"
                if args.save_zoom_images:
                    zoom_dir = os.path.join(zoom_save_root, test_split)
                    os.makedirs(zoom_dir, exist_ok=True)
                    zoom_path = os.path.join(zoom_dir, zoom_filename)
                    cropped_image.resize((args.zoom_output_width, args.zoom_output_height), resample=Image.BICUBIC).save(zoom_path)
                    zoom_image_paths.append(zoom_path)

                if args.build_zoom_dataset:
                    saved_dup_paths = save_zoom_to_dataset(
                        rebuilt_split_dir=rebuilt_split_dir,
                        cropped_image=cropped_image,
                        img_stem=img_stem,
                        turn_idx=turn_idx,
                        bbox=bbox,
                        anno_path=anno_path,
                        duplicate_times=args.zoom_duplicate_times,
                    )
                    rebuilt_dataset_zoom_image_paths.extend(saved_dup_paths)

                cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                content_f = [{"type": "text", "text": "<tool_response>"},
                             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}},
                             {"type": "text", "text": USER_PROMPT_V2},
                             {"type": "text", "text": "</tool_response>"}]
                chat_message.extend([
                    {"role": "assistant", "content": response_message},
                    {"role": "user", "content": content_f},
                ])
                print_messages.extend([
                    {"role": "assistant", "content": response_message},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                        {"type": "text", "text": USER_PROMPT_V2},
                    ]},
                ])
                turn_idx += 1
            else:
                print_messages.append({"role": "assistant", "content": response_message})

            try_count += 1
    except Exception as e:
        print("Error!!!!", e)
        status = "error"

    if "</answer>" in response_message and "<answer>" in response_message:
        output_text = response_message.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        output_text = response_message

    return {
        "image": img,
        "question": question,
        "answer": anno["options"][0],
        "pred_ans": output_text,
        "pred_output": print_messages,
        "zoom_image_paths": zoom_image_paths,
        "rebuilt_dataset_image_path": rebuilt_dataset_image_path,
        "rebuilt_dataset_zoom_image_paths": rebuilt_dataset_zoom_image_paths,
        "zoom_output_width": args.zoom_output_width,
        "zoom_output_height": args.zoom_output_height,
        "zoom_duplicate_times": args.zoom_duplicate_times,
        "status": status,
    }


if __name__ == "__main__":
    test_types = ["direct_attributes", "relative_position"]
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}_{args.result_tag}.jsonl"
        save_json = []
        test_path = os.path.join(vstar_bench_path, test_type)
        pool = multiprocessing.Pool(processes=args.num_workers)

        image_files = []
        for file_name in os.listdir(test_path):
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in valid_exts:
                continue
            anno_name = f"{os.path.splitext(file_name)[0]}.json"
            if os.path.exists(os.path.join(test_path, anno_name)):
                image_files.append(file_name)
        image_args = [[img, test_path] for img in image_files]

        with tqdm(total=len(image_args), desc="Processing V* " + test_type) as pbar:
            for result in pool.imap(process, image_args):
                if result is not None:
                    save_json.append(result)
                    pbar.update(1)

        pool.close()
        pool.join()

        with open(os.path.join(save_path, save_name), "w") as f:
            for item in save_json:
                f.write(json.dumps(item) + "\n")
