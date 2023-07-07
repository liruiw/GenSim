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

model = "gpt-4"

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_text(folder, name, out):
    mkdir_if_missing(folder)
    with open(os.path.join(folder, name + ".txt"), "w") as fhandle:
        fhandle.write(out)


def add_to_txt(full_interaction, message, with_print=False):
    """ Add the message string to the full interaction """
    full_interaction += "\n\n"+message
    if with_print:
        print("\n\n"+message)
    return full_interaction

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
    return '\n'.join(code_lines).strip(), task_name


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
    print(f"TOTAL SYNTAX_PASS_RATE: {syntax_rate * 100:.1f}% RUNTIME_PASS_RATE: {run_rate * 100:.1f}% ENV_PASS_RATE: {env_success_rate * 100:.1f}% DIVERSITE SCORE: {diversity_score:.3f}")
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

    if sample_num > 0:
        sample_idx = np.random.choice(len(task_name_dict), sample_num, replace=False)

    for idx, (task_name, task_desc) in enumerate(task_name_dict.items()):
        if idx in sample_idx:
            prompt_replacement += f'- {task_name}: {task_desc}\n'

    return prompt_replacement + "\n\n"

def sample_list_reference(item_list, sample_num=-1):
    """ sample reference code from a list of python files """
    sample_idx = list(range(len(item_list)))
    prompt_replacement = ''

    if sample_num > 0:
        sample_idx = np.random.choice(len(item_list), sample_num, replace=False)

    print("reference files: ", [item_list[idx] for idx in sample_idx])
    for idx, item in enumerate(item_list):
        item_content = open(f"cliport/tasks/{item}").read()
        if idx in sample_idx:
            prompt_replacement += f'```\n{item_content}\n```\n\n'

    return prompt_replacement + "\n\n"


def compute_diversity_score_from_assets(task_assets):
    """ compute how many new asset combos are found by a proxy"""
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