import os
import json
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import torch
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
import io
import shutil
from openai import OpenAI
import requests


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
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://10.39.19.140:8000/v1', help='API URL')
parser.add_argument('--vstar_bench_path', type=str, default=None, help='Path to the V* benchmark')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--zoom_save_path', type=str, default=None, help='Path to save zoomed crop images')
parser.add_argument('--save_zoom_images', type=str2bool, default=True, help='Whether to save zoomed crop images')
parser.add_argument('--build_zoom_dataset', type=str2bool, default=False, help='Whether to build a new dataset with original + zoomed images')
parser.add_argument('--new_dataset_suffix', type=str, default='with_zoom', help='Suffix used for rebuilt dataset folder')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()


openai_api_key = args.api_key
openai_api_base = args.api_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models['data'][0]['id']
else:
    eval_model_name = args.eval_model_name

vstar_bench_path = args.vstar_bench_path
save_path = args.save_path
save_path = os.path.join(save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)
save_zoom_images = args.save_zoom_images
build_zoom_dataset = args.build_zoom_dataset
if save_zoom_images:
    zoom_save_root = args.zoom_save_path if args.zoom_save_path else os.path.join(save_path, "zoom_crops")
    os.makedirs(zoom_save_root, exist_ok=True)
else:
    zoom_save_root = None

if build_zoom_dataset:
    vstar_abs = os.path.abspath(vstar_bench_path.rstrip(os.sep))
    vstar_parent = os.path.dirname(vstar_abs)
    vstar_name = os.path.basename(vstar_abs)
    rebuilt_dataset_root = os.path.join(vstar_parent, f"{vstar_name}_{args.new_dataset_suffix}")
    os.makedirs(rebuilt_dataset_root, exist_ok=True)
else:
    rebuilt_dataset_root = None
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

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

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""
USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

instruction_prompt_before = """Question: {question}
Options: {options}
""" + USER_PROMPT_V2

user_prompt = USER_PROMPT_V2

start_token = "<tool_call>"
end_token = "</tool_call>"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# the following code is copied from qwen-vl-utils
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
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


def process(img_arg):
    img, test_path = img_arg
    img_path = os.path.join(test_path, img)
    img_stem = os.path.splitext(img)[0]
    anno_path = os.path.join(test_path, f"{img_stem}.json")
    if not os.path.exists(anno_path):
        return None
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    question = anno['question']
    options = anno['options']

    option_str = "\n"
    for i in range(len(options)):
        option_str += abc_map[i + 1] + '. ' + options[i] + '\n'
    
    prompt = instruction_prompt_before.format(question=question, options=option_str)
    pil_img = Image.open(img_path)

    base64_image = encode_image_to_base64(img_path)

    messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    print_messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_message = messages

    response_message = ""
    zoom_image_paths = []
    rebuilt_dataset_zoom_image_paths = []
    rebuilt_dataset_image_path = None

    test_split = os.path.basename(test_path)
    rebuilt_split_dir = None
    if build_zoom_dataset:
        rebuilt_split_dir = os.path.join(rebuilt_dataset_root, test_split)
        os.makedirs(rebuilt_split_dir, exist_ok=True)
        rebuilt_img_path = os.path.join(rebuilt_split_dir, img)
        rebuilt_anno_path = os.path.join(rebuilt_split_dir, os.path.basename(anno_path))
        if not os.path.exists(rebuilt_img_path):
            shutil.copy2(img_path, rebuilt_img_path)
        if not os.path.exists(rebuilt_anno_path):
            shutil.copy2(anno_path, rebuilt_anno_path)
        rebuilt_dataset_image_path = rebuilt_img_path

    status = 'success'
    try_count = 0
    turn_idx = 0
    try:
        while '</answer>' not in response_message:
            if '</answer>' in response_message and '<answer>' in response_message:
                break

            if try_count > 10:
                break

            params = {
                "model": eval_model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 1024,
                "stop": ["<|im_end|>\n".strip()],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content
            
            if start_token in response_message:
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                action_list = eval(action_list)

                bbox_list = []
                cropped_pil_image_content_list = []

                bbox_str = action_list['arguments']['bbox_2d']
                bbox = bbox_str
                left, top, right, bottom = bbox
                cropped_image = pil_img.crop((left, top, right, bottom))
                new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                zoom_filename = f"{img_stem}_turn{turn_idx:02d}_bbox_{_bbox_token(bbox)}.png"
                if save_zoom_images:
                    zoom_dir = os.path.join(zoom_save_root, test_split)
                    os.makedirs(zoom_dir, exist_ok=True)
                    zoom_path = os.path.join(zoom_dir, zoom_filename)
                    cropped_image.save(zoom_path)
                    zoom_image_paths.append(zoom_path)
                if build_zoom_dataset:
                    rebuilt_zoom_path = os.path.join(rebuilt_split_dir, zoom_filename)
                    cropped_image.save(rebuilt_zoom_path)
                    rebuilt_dataset_zoom_image_paths.append(rebuilt_zoom_path)
                    rebuilt_zoom_anno = os.path.join(rebuilt_split_dir, f"{os.path.splitext(zoom_filename)[0]}.json")
                    if not os.path.exists(rebuilt_zoom_anno):
                        shutil.copy2(anno_path, rebuilt_zoom_anno)
                cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                bbox_list.append(bbox)
                cropped_pil_image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}}
                cropped_pil_image_content_list.append(cropped_pil_image_content)

                if len(bbox_list) == 1:
                    bbox_list = bbox_list[0]
                user_msg = user_prompt

                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                for cropped_pil_image_content in cropped_pil_image_content_list:
                    content_f.append(cropped_pil_image_content)
                content_f.append({"type": "text", "text": user_msg})
                content_f.append({"type": "text", "text": "</tool_response>"})

                _message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": content_f,
                    }
                ]

                chat_message.extend(_message)
            
                p_message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                            {"type": "text", "text": user_msg},
                        ],
                    }
                ]
                print_messages.extend(p_message)
                turn_idx += 1
            else:
                p_message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)


            try_count += 1
    except Exception as e:
        print(f"Error!!!!", e)
        status = 'error'
                

    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info['image'] = img
    save_info['question'] = question
    save_info['answer'] = anno['options'][0]
    save_info['pred_ans'] = output_text
    save_info['pred_output'] = print_messages
    save_info['zoom_image_paths'] = zoom_image_paths
    save_info['rebuilt_dataset_image_path'] = rebuilt_dataset_image_path
    save_info['rebuilt_dataset_zoom_image_paths'] = rebuilt_dataset_zoom_image_paths
    save_info['status'] = status
    return save_info


if __name__ == "__main__":
    test_types = ['direct_attributes', 'relative_position']
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}.jsonl"
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

        with tqdm(total=len(image_args), desc="Processing V* "+test_type) as pbar:
            for result in pool.imap(process, image_args):
                if result is not None:
                    save_json.append(result)
                    pbar.update(1)

        pool.close()
        pool.join()
    
        with open(os.path.join(save_path, save_name), 'w') as f:
            for item in save_json:
                f.write(json.dumps(item) + '\n')
