# coding=utf-8
# Copyright 2022 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data collection script."""

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
from cliport.simgen_utils import (mkdir_if_missing,
        save_text,
        add_to_txt,
        extract_code,
        extract_dict,
        extract_list,
        extract_assets,
        format_dict_prompt,
        sample_list_reference,
        save_stat,
        compute_diversity_score_from_assets)



openai.api_key = "YOUR_KEY"
model = "gpt-4"
NEW_TASK_LIST = []
full_interaction = ''

def generate_feedback(prompt, max_tokens=2048, temperature=0.0, model="gpt-4", assistant_prompt=None, interaction_txt=None):
    """ use GPT-4 API """
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
                  {"role": "user", "content": prompt}],
    }
    if assistant_prompt is not None:
        params["messages"].append({"role": "assistant", "content": assistant_prompt})

    for retry in range(3):
        try:
            if interaction_txt is not None:
                interaction_txt = add_to_txt(interaction_txt, ">>> Prompt: \n" + prompt, with_print=False)
            res = openai.ChatCompletion.create(**params)["choices"][0]["message"]["content"]
            to_print = highlight(f"{res}", PythonLexer(), TerminalFormatter())
            print(to_print)
            if interaction_txt is not None:
                interaction_txt = add_to_txt(interaction_txt,  ">>> Answer: \n" + res, with_print=False)
                return res, interaction_txt
            return res

        except Exception as e:
            print("failed chat completion", e)
    raise Exception("Failed to generate")


def llm_gen_env(cfg, model_output_dir):
    """
    The LLM running pipeline
    """
    global full_interaction
    start_time = time.time()
    prompt_folder = f"prompts/{cfg['prompt_folder']}"
    task_prompt_text = open(f"{prompt_folder}/cliport_prompt_task.txt").read()
    res, full_interaction = generate_feedback(task_prompt_text, temperature=cfg['gpt_temperature'], interaction_txt=full_interaction)

    # Extract dictionary for task name, descriptions, and assets
    task_def = extract_dict(res, prefix="new_task")
    exec(task_def, globals())

    full_interaction = add_to_txt(full_interaction, "================= Task and Asset Design!", with_print=True)
    pprint(new_task)
    save_text(model_output_dir, f'{new_task["task-name"]}_task_def_output', res)

    # Asset Generation
    if os.path.exists(f"{prompt_folder}/cliport_prompt_asset_template.txt"):
        full_interaction = add_to_txt(full_interaction, "================= Asset Generation!", with_print=True)
        asset_prompt_text = open(f'{prompt_folder}/cliport_prompt_asset_template.txt').read()
        asset_prompt_text = asset_prompt_text.replace("TASK_NAME_TEMPLATE", new_task["task-name"])
        asset_prompt_text = asset_prompt_text.replace("ASSET_STRING_TEMPLATE", str(new_task["assets-used"]))

        res, full_interaction = generate_feedback(asset_prompt_text, temperature=0, assistant_prompt=res, interaction_txt=full_interaction) # cfg['gpt_temperature']
        save_text(model_output_dir,  f'{new_task["task-name"]}_asset_output', res)
        asset_list = extract_assets(res)
        # save_urdf(asset_list)
    else:
        asset_list = {}

    # API Preview
    if os.path.exists(f"{prompt_folder}/cliport_prompt_api_template.txt"):
        full_interaction = add_to_txt(full_interaction,"================= API Preview!")
        api_prompt_text = open(f'{prompt_folder}/cliport_prompt_api_template.txt').read()
        api_prompt_text = api_prompt_text.replace("TASK_NAME_TEMPLATE", new_task["task-name"])
        res, full_interaction = generate_feedback(api_prompt_text, temperature=0, assistant_prompt=res, interaction_txt=full_interaction) # cfg['gpt_temperature']

    # Error Preview
    if os.path.exists(f"{prompt_folder}/cliport_prompt_common_errors_template.txt"):
        full_interaction = add_to_txt(full_interaction,"================= Error Book Preview!")
        errorbook_prompt_text = open(f'{prompt_folder}/cliport_prompt_common_errors_template.txt').read()
        errorbook_prompt_text = errorbook_prompt_text.replace("TASK_NAME_TEMPLATE", new_task["task-name"])
        res, full_interaction = generate_feedback(errorbook_prompt_text, temperature=0., assistant_prompt=res, interaction_txt=full_interaction) # cfg['gpt_temperature']

    # Generate Code
    if os.path.exists(f"{prompt_folder}/cliport_prompt_code_split_template.txt"):
        full_interaction = add_to_txt(full_interaction,"================= Code Generation!")
        code_prompt_text = open(f"{prompt_folder}/cliport_prompt_code_split_template.txt").read()
        code_prompt_text = code_prompt_text.replace("TASK_NAME_TEMPLATE", new_task["task-name"])
        code_prompt_text = code_prompt_text.replace("TASK_STRING_TEMPLATE", str(new_task))
        res, full_interaction = generate_feedback(code_prompt_text, temperature=0., assistant_prompt=res, interaction_txt=full_interaction) # cfg['gpt_temperature']

    code, task_name = extract_code(res)

    if len(task_name) == 0:
        print("empty task name:", task_name)
        return None

    save_text(model_output_dir, task_name + '_code_output', code)
    try:
        exec(code, globals())
    except:
        print(str(traceback.format_exc()))
        return None

    cfg['task'] = new_task["task-name"]
    print("save all interaction to :", f'{new_task["task-name"]}_full_output')
    save_text(model_output_dir, f'{new_task["task-name"]}_full_output', full_interaction)
    print(f"\n\nLLM generation time: {time.time() - start_time}")
    return task_name, new_task, asset_list, code


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    global full_interaction

    # Evaluation Metric
    SYNTAX_PASS_RATE = 0.
    RUNTIME_PASS_RATE = 0.
    ENV_PASS_RATE = 0.
    DIVERSITY_SCORES = 0

    task_assets = []
    start_time = time.time()
    output_folder = 'output/output_stats'

    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    model_output_dir = os.path.join(output_folder, cfg['prompt_folder'] + "_" + model_time)
    TOTAL_TRIALS = cfg['trials']
    env_names = []

    for trial_i in range(TOTAL_TRIALS):

        # generate
        res = llm_gen_env(cfg, model_output_dir)
        if res is not None:
            SYNTAX_PASS_RATE += 1
            task_name, new_task, asset_list, code = res
            task_assets.append(new_task["assets-used"])
            env_names.append(task_name)
        else:
            env_names.append("")
            print("Syntax Failure")
            continue

        try:
            env = Environment(
                cfg['assets_root'],
                disp=cfg['disp'],
                shared_memory=cfg['shared_memory'],
                hz=480,
                record_cfg=cfg['record']
            )

            task = eval(task_name)()
            task.mode = cfg['mode']
            record = cfg['record']['save_video']
            save_data = cfg['save_data']

            # Initialize scripted oracle agent and dataset.
            agent = task.oracle(env)
            data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
            dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
            print(f"Saving to: {data_path}")
            print(f"Mode: {task.mode}")

            # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
            seed = dataset.max_seed
            total_cnt = 0.
            reset_success_cnt = 0.
            env_success_cnt = 0.

            # Start video recording (NOTE: super slow)
            if record:
                env.start_rec(f'{dataset.n_episodes+1:06d}')

            # Collect training data from oracle demonstrations.
            # while dataset.n_episodes < cfg['n']:
            while total_cnt < cfg['max_env_run_cnt']:
                total_cnt += 1
                if total_cnt == cfg['max_env_run_cnt'] or total_cnt == cfg['n']:
                    if reset_success_cnt == total_cnt - 1:
                        RUNTIME_PASS_RATE += 1
                        print("Runtime Test Pass!")

                        # the task can actually be completed with oracle
                        if env_success_cnt >= total_cnt / 2:
                            ENV_PASS_RATE += 1
                            print("Environment Test Pass!")
                        else:
                            print("Bad task design!! Reset!")

                    break

                episode, total_reward = [], 0
                seed += 2

                # Set seeds.
                np.random.seed(seed)
                random.seed(seed)
                print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))
                env.set_task(task)

                try:
                    obs = env.reset()
                except Exception as e:
                    print("reset exception:", str(traceback.format_exc()))
                    continue

                info = env.info
                reward = 0


                # Rollout expert policy
                for _ in range(task.max_steps):
                    act = agent.act(obs, info)
                    episode.append((obs, act, reward, info))
                    lang_goal = info['lang_goal']
                    obs, reward, done, info = env.step(act)
                    total_reward += reward
                    print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
                    if done:
                        break

                episode.append((obs, None, reward, info))

                # End video recording
                if record:
                    env.end_rec()

                # Only save completed demonstrations.
                if save_data and total_reward > 0.99:
                    dataset.add(seed, episode)

                reset_success_cnt += 1
                env_success_cnt += total_reward > 0.99

            p.disconnect()

        except:
            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
            save_text(model_output_dir, task_name + '_error', str(traceback.format_exc()))

            print("========================================================")
            print("Exception:", to_print)
            p.disconnect()

        print("=========================================================")
        print(f"SYNTAX_PASS_RATE: {(SYNTAX_PASS_RATE / (trial_i+1)) * 100:.1f}% RUNTIME_PASS_RATE: {(RUNTIME_PASS_RATE / (trial_i+1)) * 100:.1f}% ENV_PASS_RATE: {(ENV_PASS_RATE / (trial_i+1)) * 100:.1f}%")
        print("=========================================================")

        prompt_folder = f"prompts/{cfg['prompt_folder']}"
        if os.path.exists(f"{prompt_folder}/cliport_prompt_task_reflection.txt") and env_success_cnt >= 1:
            # only consider successful task
            full_interaction = add_to_txt(full_interaction,"================= Code Reflect!")

            base_task_path = os.path.join("prompts/data", 'base_tasks.json')
            base_tasks = json.load(open(base_task_path))

            # append current new task
            for task in NEW_TASK_LIST:
                base_tasks[task["task-name"].replace("-", "_")] = str(task)

            task_descriptions_replacement_str = format_dict_prompt(base_tasks, -1)
            code_reflection_prompt_text = open(f"{prompt_folder}/cliport_prompt_task_reflection.txt").read()
            code_reflection_prompt_text = code_reflection_prompt_text.replace("CURRENT_TASK_NAME_TEMPLATE", str(task_descriptions_replacement_str))
            code_reflection_prompt_text = code_reflection_prompt_text.replace("TASK_STRING_TEMPLATE", str(new_task))
            res, full_interaction = generate_feedback(code_reflection_prompt_text, temperature=0., interaction_txt=full_interaction) # cfg['gpt_temperature']
            reflection_def_cmd = extract_dict(res, prefix='task_reflection')
            exec(reflection_def_cmd, globals())
            print("save task result:", task_reflection)

            if task_reflection["add_to_the_task_list"] == 'True':
                NEW_TASK_LIST.append(new_task)

                if cfg['save_memory']:
                    print("actually saving!")

                    # write the python file and append to the task descriptions
                    generated_task_code_path = os.path.join(cfg['prompt_data_path'], 'generated_task_codes.json')
                    generated_task_codes = json.load(open(generated_task_code_path))
                    generated_task_codes.append(new_task["task-name"] + ".py")
                    with open('cliport/generated_tasks/' + new_task["task-name"].replace("-","_") + ".py", "w") as fhandle:
                        fhandle.write(code)

                    with open(generated_task_code_path, "w") as outfile:
                        json.dump(generated_task_codes, outfile, indent=4)

                    generated_task_path = os.path.join(cfg['prompt_data_path'], 'generated_tasks.json')
                    generated_tasks = json.load(open(generated_task_path))
                    generated_tasks[new_task["task-name"]] = new_task

                    with open(generated_task_path, "w") as outfile:
                        json.dump(generated_tasks, outfile, indent=4)

    print("task_assets:", task_assets)
    DIVERSITY_SCORE = compute_diversity_score_from_assets(task_assets)
    save_stat(cfg, model_output_dir, env_names, SYNTAX_PASS_RATE / TOTAL_TRIALS, RUNTIME_PASS_RATE / TOTAL_TRIALS, ENV_PASS_RATE / TOTAL_TRIALS, DIVERSITY_SCORE)
    print(f"Total {len(NEW_TASK_LIST)} New Added Tasks:", NEW_TASK_LIST)

if __name__ == '__main__':
    main()
