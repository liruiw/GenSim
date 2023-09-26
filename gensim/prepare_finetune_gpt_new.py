import cv2
import numpy as np
import IPython
import os

import openai
import pandas as pd
import json
import subprocess
from gensim.utils import set_gpt_model, clear_messages, format_finetune_prompt, format_finetune_prompt_codeonly


def format_completion_codeonly(task_name, descriptions, code):
    completion_text = " \n```python\n" + code  + "\n```\n\nSTOP"
    return completion_text

def format_completion(task_name, descriptions, code):
    completion_text = f" \n {task_name}: {descriptions}```\n\n###"
    completion_text += "\n```python\n" + code  + "\n```\n\nSTOP"
    return completion_text

# test if using the finetuned model can generate better task coed than the base model
# https://platform.openai.com/docs/guides/fine-tuning
data_path = 'prompts/data'
def load_offline_memory():
    """get the current task descriptions, assets, and code"""
    base_task_path = os.path.join(data_path, "base_tasks.json")
    base_asset_path = os.path.join(data_path, "base_assets.json")
    base_task_code_path = os.path.join(data_path, "base_task_codes.json")

    base_tasks = json.load(open(base_task_path))
    base_assets = json.load(open(base_asset_path))
    base_task_codes = json.load(open(base_task_code_path))

    generated_task_path = os.path.join(data_path, "generated_tasks.json")
    generated_asset_path = os.path.join(data_path, "generated_assets.json")
    generated_task_code_path = os.path.join(data_path, "generated_task_codes.json")

    # print("original base task num:", len(base_tasks))
    base_tasks.update(json.load(open(generated_task_path)))
    # base_assets.update(json.load(open(generated_asset_path)))

    for task in json.load(open(generated_task_code_path)):
        if task not in base_task_codes:
            base_task_codes.append(task)

        # print("current base task num:", len(base_tasks))
    return base_tasks, base_assets, base_task_codes


code_buffer = {}
base_tasks, base_assets, base_task_codes = load_offline_memory()
TOTAL_DATASET_TOKENS = 0

added_tasks = []
df = pd.DataFrame()
file_name = 'prompts/finetune_data_new.jsonl'
file = open(file_name, 'w')


for task_file in base_task_codes:
    ## TODO(lirui): consider adding more structure here.
    task_name = task_file[:-3].replace("_", "-")
    if task_name in added_tasks:
        continue

    if task_name not in base_tasks:
        print(f"{task_name} missing")
        continue

    added_tasks.append(task_name)
    task_description = base_tasks[task_name]

    if os.path.exists("cliport/tasks/" + task_file):
        task_code = open("cliport/tasks/" + task_file).read()

        # the generated cliport task path
    elif os.path.exists("cliport/generated_tasks/" + task_file):
        task_code = open("cliport/generated_tasks/" + task_file).read()

    # prompt = format_finetune_prompt(task_name)
    # completion = format_completion(task_name, task_description, task_code)

    prompt = format_finetune_prompt_codeonly(task_name)
    completion = format_completion_codeonly(task_name, task_description, task_code)


    # rough estimates
    TOTAL_DATASET_TOKENS += len(prompt) / 4
    TOTAL_DATASET_TOKENS += len(completion) / 4
    # new_row = { 'prompt': prompt,
    #             'completion': completion}
    # new_row = {"role": "system", "content": "You are an AI in robot simulation code and task design."}
    # new_row = pd.DataFrame([new_row])
    # df = pd.concat([df, new_row], axis=0, ignore_index=True)

    # new_row = {"role": "user", "content": prompt}
    # new_row = pd.DataFrame([new_row])
    # df = pd.concat([df, new_row], axis=0, ignore_index=True)

    # new_row = {"role": "assistant", "content": completion}
    # new_row = pd.DataFrame([new_row])
    # df = pd.concat([df, new_row], axis=0, ignore_index=True)
    data = ({"messages": [{"role": "system", "content": "You are an AI in robot simulation code and task design."},
                         {"role": "user", "content": prompt},
                         {"role": "assistant", "content": completion}]})
    # print(data)
    file.write(json.dumps(data)+"\n") # jsonl

# df.to_csv("prompts/finetune_data.csv",index=False)
print("======================================")
print("estimate number of tokens:", TOTAL_DATASET_TOKENS)
print("estimate price for davinci:", TOTAL_DATASET_TOKENS / 1000 * 0.03)
print("total number of instructions:", len(df))
print("======================================")
# actual finetuning

## prepared_data.csv --> prepared_data_prepared.json
# subprocess.run('openai tools fine_tunes.prepare_data --file prompts/finetune_data.csv --quiet'.split())

print("now you can run \n  python misc/job_create.py")
print("check file!:", file_name)

# Model    Training    Usage
# Ada $0.0004 / 1K tokens $0.0016 / 1K tokens
# Curie   $0.0030 / 1K tokens $0.0120 / 1K tokens
# Davinci $0.0300 / 1K tokens $0.1200 / 1K tokens

# ## Start fine-tuning
# openai api fine_tunes.create --training_file prompts/finetune_data_new.jsonl --model gpt-3.5-turbo --suffix "GenSimNew"
# subprocess.run('openai api fine_tunes.create --training_file output/finetune_data_prepared.jsonl --model davinci --suffix "GenSim"'.split())


# Tracking Finetune Status
# openai api fine_tunes.follow -i
# openai api fine_tunes.get -i
# openai wandb sync