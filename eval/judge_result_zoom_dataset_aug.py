import os
import json
import numpy as np
import multiprocessing
import argparse
from tqdm import tqdm
from openai import OpenAI
import requests

multiprocessing.set_start_method("spawn", force=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="qwen", help="Model name for result save")
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument("--api_url", type=str, default="http://10.39.19.140:8000/v1", help="API URL")
parser.add_argument("--vstar_bench_path", type=str, default=None, help="Path to the V* benchmark")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the results")
parser.add_argument("--result_tag", type=str, default="zoom_aug", help="Result file suffix tag")
parser.add_argument("--eval_model_name", type=str, default=None, help="Model name for evaluation")
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()

client = OpenAI(api_key=args.api_key, base_url=args.api_url)
if args.eval_model_name is None:
    response = requests.get(f"{args.api_url}/models")
    models = response.json()
    eval_model_name = models["data"][0]["id"]
else:
    eval_model_name = args.eval_model_name

test_types = ["direct_attributes", "relative_position"]


def get_chat_template():
    return """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""


def get_gpt4_score_ice():
    return [
        """
[Question]: Is the countertop tan or blue?
[Standard Answer]: A. The countertop is tan.
[Model_answer] : tan
Judgement: 1
""",
        """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: A. The barrier is on the left side of the picture.
[Model_answer] : A
Judgement: 1
""",
        """
[Question]: Is the kite brown and large?
[Standard Answer]: A. Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""",
        """
[Question]: Are the spots on a giraffe?
[Standard Answer]: A. No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""",
        """
[Question]: Who is wearing pants?
[Standard Answer]: A. The boy is wearing pants.
[Model_answer] : C. The girl in the picture is wearing pants.
Judgement: 0
""",
    ]


def get_prompt(predict_str, ground_truth, question):
    prompt = get_chat_template()
    for example in get_gpt4_score_ice():
        prompt += example + "\n\n"
    prompt += f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    return prompt


result_root_path = os.path.join(args.save_path, args.model_name)
all_acc = []
per_type_acc = {k: [] for k in test_types}
error_nums_type = {k: 0 for k in test_types}


def process(line):
    data = json.loads(line.strip())
    question = data["question"]
    answer = "A. " + data["answer"]
    pred_ans = data["pred_ans"]

    if "\\boxed" in pred_ans:
        pred_ans = pred_ans.split("\\boxed{")[1].split("}")[0]

    acc_reward = 0.0
    if len(pred_ans) == 1:
        acc_reward = 1.0 if pred_ans == "A" else 0.0
    elif len(pred_ans) == 2 and "." in pred_ans:
        acc_reward = 1.0 if "A" in pred_ans else 0.0
    elif answer in pred_ans:
        acc_reward = 1.0
    else:
        full_prompt = get_prompt(pred_ans, answer, question)
        response = client.chat.completions.create(
            model=eval_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.0,
        ).choices[0].message.content.strip()

        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()
        if response == "1" or "1" in response:
            acc_reward = 1.0
        elif response == "0" or "0" in response:
            acc_reward = 0.0
        else:
            acc_reward = 0.0
    return acc_reward, data


if __name__ == "__main__":
    error_preds = []

    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}_{args.result_tag}.jsonl"
        result_path = os.path.join(result_root_path, save_name)
        save_json = []
        pool = multiprocessing.Pool(processes=args.num_workers)
        with open(result_path, "r") as f:
            lines = f.readlines()

        with tqdm(total=len(lines), desc="Judging V* " + test_type) as pbar:
            for result in pool.imap(process, lines):
                if result is not None:
                    acc_reward, data = result
                    all_acc.append(acc_reward)
                    per_type_acc[test_type].append(acc_reward)
                    if acc_reward != 1.0:
                        error_preds.append({
                            "pred_ans": data["pred_ans"],
                            "question": data["question"],
                            "answer": data["answer"],
                        })
                    data["acc"] = acc_reward
                    save_json.append(data)
                    pbar.update(1)

        pool.close()
        pool.join()

        with open(os.path.join(result_root_path, save_name.replace(".jsonl", "_acc.jsonl")), "w") as f:
            for item in save_json:
                f.write(json.dumps(item) + "\n")

    final_acc = {}
    for test_type in test_types:
        final_acc[test_type] = np.mean(per_type_acc[test_type]) * 100
        print(f"Accuracy for {test_type}: {final_acc[test_type]:.2f}%")
    final_acc["overall"] = np.mean(all_acc) * 100
    print(f"Overall Accuracy: {final_acc['overall']:.2f}%")
    final_acc["error_nums"] = error_nums_type
    final_acc["error_preds"] = error_preds

    with open(os.path.join(result_root_path, f"final_acc_{args.result_tag}.json"), "w") as f:
        json.dump(final_acc, f, indent=4)
