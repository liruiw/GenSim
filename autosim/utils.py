import os

import numpy as np
import os
import hydra
import numpy as np
import random

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
import re

import openai
import IPython
import time
import pybullet as p
import traceback
from datetime import datetime
from pprint import pprint
import cv2
import re
import random
import json
import operator
import csv
import itertools

model = "gpt-4"
# model = "gpt-3.5-turbo-16k"
# model = "gpt-4-0613"

def set_gpt_model(gpt_model_name):
    """ globally set gpt-model"""
    global model
    model = gpt_model_name
    print("use gpt model:", model)

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_text(folder, name, out):
    mkdir_if_missing(folder)
    with open(os.path.join(folder, name + ".txt"), "w") as fhandle:
        fhandle.write(out)


def add_to_txt(full_interaction, message, with_print=False):
    """ Add the message string to the full interaction """
    full_interaction.append("\n\n"+message)
    if with_print:
        print("\n\n"+message)
    return full_interaction

def get_task_import_str():
    return "import numpy as np\n" + \
    "import os\n" + \
    "import pybullet as p\n" + \
    "import random\n" + \
    "from cliport.tasks import primitives\n" + \
    "from cliport.tasks.grippers import Spatula\n" + \
    "from cliport.tasks.task import Task\n" + \
    "from cliport.utils import utils\n"

def extract_code(res):
    """ parse code block """
    # Pattern to find string between ```
    pattern = r'```(.*?)```'

    # Use re.findall to get all substrings within ```
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
        print("\n".join(res.split("\n")))
        print("empty code string")
        return '', ''

    code_string = code_string[0]
    code_string = code_string.replace('python', '')
    code_lines = code_string.split("\n")

    if 'python' in code_string:
        code_lines = code_lines[1:] # skip the first line

    class_def = [line for line in code_lines if line.startswith('class')]
    task_name = class_def[0]
    task_name = task_name[task_name.find("class "): task_name.rfind("(Task)")][6:]

    print("task_name:", task_name)
    return get_task_import_str() + '\n'.join(code_lines).strip(), task_name


def extract_dict(res, prefix="new_task"):
    """ parse task dictionary """
    pattern = r'{(.*?)}'
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
      return ''

    code_string = code_string[0]
    code_string = code_string.replace('python', '')

    return prefix + '={'+ code_string.replace("\n","").strip() + '}'

def extract_list(res, prefix="code_reference"):
    """ parse task dictionary """
    pattern = r'\[(.*?)\]'
    code_string = re.findall(pattern, res, re.DOTALL)
    if len(code_string) == 0:
      return ''

    code_string = code_string[0]
    return prefix + '=[' + code_string.strip() + ']'

def extract_assets(res):
    """ parse generated assets """
    pattern = r'<?xml(.*?)</robot>'
    code_string = re.findall(pattern, res, re.DOTALL)

    assets_pattern = r'robot name="(.*?)">'
    assets_string = re.findall(assets_pattern, res, re.DOTALL)
    if len(code_string) == 0:
        return {}

    try:
        new_urdf = {}
        for asset_path, code in zip(assets_string, code_string):
            new_urdf[asset_path] = "<?xml"+code

        # new_urdf_cmd ='new_urdf={' + code_string[0].rstrip() + '}'
        # exec(new_urdf_cmd)
        return new_urdf

    except:
        print("asset creation failure")
        print(str(traceback.format_exc()))
        return None

def save_stat(cfg, output_dir, env_names, syntax_rate, run_rate, env_success_rate, diversity_score):
    """ save run results """
    print("=========================================================")
    print(f"{cfg['prompt_folder']} | TOTAL SYNTAX_PASS_RATE: {syntax_rate * 100:.1f}% RUNTIME_PASS_RATE: {run_rate * 100:.1f}% ENV_PASS_RATE: {env_success_rate * 100:.1f}% DIVERSITE SCORE: {diversity_score:.3f}")
    print("=========================================================")

    with open(os.path.join(output_dir, "eval_results.csv"), "w") as f:
        writer = csv.writer(f)
        row_info_name = ["prompt", "metric", "success"]
        writer.writerow(row_info_name)
        for col, stat in zip(["syntax", "runtime", "env. completion", "diversity"], [syntax_rate, run_rate, env_success_rate, diversity_score]):
            row_info = [cfg['prompt_folder'], col, stat]
            writer.writerow(row_info)

def format_dict_prompt(task_name_dict, sample_num=-1, sort_items=False):
    """ format a saved dictionary into prompt """
    if sort_items:
        task_name_dict = sorted(task_name_dict.items(), key=operator.itemgetter(0))
    prompt_replacement = ''
    sample_idx = list(range(len(task_name_dict)))
    random.shuffle(sample_idx)

    if sample_num > 0:
        sample_idx = np.random.choice(len(task_name_dict), sample_num, replace=False)

    for idx, (task_name, task_desc) in enumerate(task_name_dict.items()):
        if idx in sample_idx:
            prompt_replacement += f'- {task_name}: {task_desc}\n'

    return prompt_replacement + "\n\n"

def format_list_prompt(task_list, sample_num=-1, sort_items=False):
    """ format a saved dictionary into prompt """

    # if sort_items:
    #     task_list = sorted(task_list, key=operator.itemgetter(0))
    prompt_replacement = ''
    sample_idx = list(range(len(task_list)))

    if sample_num > 0:
        sample_idx = np.random.choice(len(task_list), sample_num, replace=False)

    for idx, task in enumerate(task_list):
        if idx in sample_idx:
            prompt_replacement += f"- {task['task-name']}: {task['task-descriptions']}\n"

    return prompt_replacement + "\n\n"

def sample_list_reference(item_list, sample_num=-1):
    """ sample reference code from a list of python files """
    sample_idx = list(range(len(item_list)))
    prompt_replacement = ''

    if sample_num > 0:
        sample_idx = np.random.choice(len(item_list), sample_num, replace=False)

    print("reference files: ", [item_list[idx] for idx in sample_idx])
    for idx, item in enumerate(item_list):
        try:
            item_content = open(f"cliport/tasks/{item}").read()
        except:
            # one or the other
            item_content = open(f"cliport/generated_tasks/{item}").read()

        if idx in sample_idx:
            prompt_replacement += f'```\n{item_content}\n```\n\n'

    return prompt_replacement + "\n\n"


def compute_diversity_score_from_assets_old(task_assets):
    """ compute how many new asset combos are covered by previous by a proxy"""
    if len(task_assets) == 0:
        return 0

    existing_assets = []
    for asset in task_assets:
        new_asset_flag = True
        for existing_asset in existing_assets:
            # it's covered by any previous assets
            if set(asset).issubset(existing_asset):
                new_asset_flag = False
                break

        if new_asset_flag:
            existing_assets.append(asset)

    return len(existing_assets) / len(task_assets)

def iou_assets(asset1, asset2):
    asset1 = set(asset1)
    asset2 = set(asset2)
    return len(asset1 & asset2) / len(asset1 | asset2)

def compute_diversity_score_from_assets(task_assets, total_trials):
    """ compute the pairwise IOU for assets"""
    if len(task_assets) == 0:
        return 0

    score = 0
    pairs = list(itertools.combinations(range(len(task_assets)), 2))
    for j, k in pairs:
        score += 1. - iou_assets(task_assets[j], task_assets[k])

    return score / len(pairs)

def truncate_message_for_token_limit(message_history, max_tokens=6000):
    truncated_messages = []
    tokens = 0

    # reverse
    for idx in range(len(message_history)-1, -1, -1) :
        message = message_history[idx]
        message_tokens = len(message['content']) / 4 # rough estimate.
        # print("message_tokens:", message['content'])
        if tokens + message_tokens > max_tokens:
            break  # This message would put us over the limit

        truncated_messages.append(message)
        tokens += message_tokens

    truncated_messages.reverse()
    # print("truncated messages:", len(truncated_messages))
    return truncated_messages

def insert_system_message(message_history):
    system_message_prompt = 'You are a helpful and expert assistant in robot simulation code writing and task design.'
    'You design tasks that are creative and do-able by table-top manipulation. '
    'You write code without syntax errors and always think through and document your code carefully. '
    message_history.insert(0, {"role": "system", "content": system_message_prompt})

# globally always feed the previous reply as the assistant message back into the model
existing_messages = []
def generate_feedback(prompt, max_tokens=2048, temperature=0.0, interaction_txt=None, retry_max=5, n=1):
    """ use GPT-4 API """
    global existing_messages
    existing_messages.append({"role": "user", "content": prompt})
    truncated_messages = truncate_message_for_token_limit(existing_messages)
    insert_system_message(truncated_messages)

    params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": truncated_messages,
        "n": n
    }

    for retry in range(retry_max):
        try:
            if interaction_txt is not None:
                add_to_txt(interaction_txt, ">>> Prompt: \n" + prompt, with_print=False)
            call_res = openai.ChatCompletion.create(**params)
            res = call_res["choices"][0]["message"]["content"]
            existing_messages.append({"role": "assistant", "content": res})

            to_print = highlight(f"{res}", PythonLexer(), TerminalFormatter())
            print(to_print)
            if interaction_txt is not None:
                add_to_txt(interaction_txt,  ">>> Answer: \n" + res, with_print=False)

            if n > 1:
                return [r["message"]["content"] for r in call_res["choices"]]
            return res

        except Exception as e:
            print("failed chat completion", e)
    raise Exception("Failed to generate")

def clear_messages():
    global existing_messages
    existing_messages = []