import os
import json
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
from tqdm import tqdm
import base64
from openai import OpenAI
import requests


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://10.39.19.140:8000/v1', help='API URL')
parser.add_argument('--vstar_bench_path', type=str, default=None, help='Path to the V* benchmark')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--result_tag', type=str, default='bbox_only', help='Suffix tag for result file names')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()


openai_api_key = args.api_key
openai_api_base = args.api_url
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models['data'][0]['id']
else:
    eval_model_name = args.eval_model_name

save_path = os.path.join(args.save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)

abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
instruction_prompt_system = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Locate a region using a 2D bbox.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box [x1, y1, x2, y2]."},"label":{"type":"string","description":"Optional label."}},"required":["bbox_2d"]}}}
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
        return base64.b64encode(image_file.read()).decode('utf-8')


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
    status = 'success'
    try_count = 0
    bbox_history = []
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

            if start_token in response_message and end_token in response_message:
                action_str = response_message.split(start_token)[1].split(end_token)[0].strip()
                action = eval(action_str)
                bbox = action['arguments']['bbox_2d']
                bbox_history.append(bbox)

                tool_text = (
                    "<tool_response>\n"
                    f"BBOX: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n"
                    "Use this coordinate region to continue reasoning.\n"
                    f"TOOL_TURN: {turn_idx}\n"
                    "</tool_response>"
                )

                chat_message.extend([
                    {"role": "assistant", "content": response_message},
                    {"role": "user", "content": [{"type": "text", "text": tool_text}]},
                ])
                print_messages.extend([
                    {"role": "assistant", "content": response_message},
                    {"role": "user", "content": [{"type": "text", "text": tool_text}]},
                ])
                turn_idx += 1
            else:
                print_messages.append({"role": "assistant", "content": response_message})

            try_count += 1
    except Exception as e:
        print("Error!!!!", e)
        status = 'error'

    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message

    return {
        'image': img,
        'question': question,
        'answer': anno['options'][0],
        'pred_ans': output_text,
        'pred_output': print_messages,
        'bbox_history': bbox_history,
        'tool_mode': 'bbox_only',
        'status': status,
    }


if __name__ == "__main__":
    test_types = ['direct_attributes', 'relative_position']
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}_{args.result_tag}.jsonl"
        save_json = []
        test_path = os.path.join(args.vstar_bench_path, test_type)
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

        with open(os.path.join(save_path, save_name), 'w') as f:
            for item in save_json:
                f.write(json.dumps(item) + '\n')
