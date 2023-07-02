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

openai.api_key = "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"
model = "gpt-4"
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


def save_base_memory(data_path, new_task, asset_list, code, cfg):
    """ save the current task descriptions, assets, and code, if it passes reflection and environment test """
    print("in save base memory")
    IPython.embed()
    prompt_folder = f"prompts/{cfg['prompt_folder']}"
    base_task_path = os.path.join(data_path, 'base_tasks.json')
    base_tasks = json.load(open(base_task_path))

    task_descriptions_replacement_str = format_dict_prompt(base_tasks, -1)
    code_reflection_prompt_text = open(f"{prompt_folder}/cliport_prompt_task_reflection.txt").read()
    code_reflection_prompt_text = code_reflection_prompt_text.replace("CURRENT_TASK_NAME_TEMPLATE", str(task_descriptions_replacement_str))
    code_reflection_prompt_text = code_reflection_prompt_text.replace("TASK_STRING_TEMPLATE", str(new_task))
    res, full_interaction = generate_feedback(code_reflection_prompt_text, temperature=0., interaction_txt=full_interaction) # cfg['gpt_temperature']
    reflection_def_cmd = extract_dict(res, prefix='task_reflection')
    exec(reflection_def_cmd, globals())
    print("save task result:", task_reflection)

    if task_reflection["add_to_the_task_list"] == 'False':
        return

    print("Saving new task descriptions, assets, and code!")
    generated_task_path = os.path.join(data_path, 'generated_tasks.json')
    generated_asset_path = os.path.join(data_path, 'generated_assets.json')
    generated_task_code_path = os.path.join(data_path, 'generated_task_codes.json')

    print("original base task num:", len(base_tasks))
    generated_tasks = json.load(open(generated_task_path))
    generated_assets = json.load(open(generated_asset_path))
    generated_task_codes = json.load(open(generated_task_code_path))
    generated_tasks[new_task["task-name"]] = new_task

    if False:
        # code save in cliport/generated_tasks and assets saved in cliport/environments/generated_assets
        generated_task_codes.append(new_task["task-name"] + ".py")
        print("Save code")
        with open('cliport/generated_tasks/' + new_task["task-name"] + ".py", "w") as fhandle:
            fhandle.write(code)

        # save urdf
        print("Save URDF")
        for (file_path, urdf_str) in asset_list.items():
            generated_assets.append(file_path)
            full_path = 'cliport/environments/generated_assets/' + file_path
            dir_path = os.path.dirname(full_path)
            mkdir_if_missing(dir_path)

            if os.path.exists(full_path):
                print(f"{full_path} already exists!")
            else:
                with open(full_path, "w") as fhandle:
                    fhandle.write(urdf_str)

        with open(generated_task_path, "w") as outfile:
            json.dump(generated_tasks, outfile, indent=4)

        with open(generated_asset_path, "w") as outfile:
            json.dump(generated_assets, outfile, indent=4)

        with open(generated_task_code_path, "w") as outfile:
            json.dump(generated_task_codes, outfile, indent=4)


def get_base_memory(data_path, cfg):
    """ get the current task descriptions, assets, and code """
    base_task_path = os.path.join(data_path, 'base_tasks.json')
    base_asset_path = os.path.join(data_path, 'base_assets.json')
    base_task_code_path = os.path.join(data_path, 'base_task_codes.json')

    base_tasks = json.load(open(base_task_path))
    base_assets = json.load(open(base_asset_path))
    base_task_codes = json.load(open(base_task_code_path))

    if cfg['load_memory']:
        generated_task_path = os.path.join(data_path, 'generated_tasks.json')
        generated_asset_path = os.path.join(data_path, 'generated_assets.json')
        generated_task_code_path = os.path.join(data_path, 'generated_task_codes.json')

        print("original base task num:", len(base_tasks))
        base_tasks += json.load(open(generated_task_path))
        base_assets.update(json.load(open(generated_asset_path)))
        base_task_codes += json.load(open(generated_task_code_path))
        print("current base task num:", len(base_tasks))
    return base_tasks, base_assets, base_task_codes

def llm_gen_env(cfg, model_output_dir):
    """
    The LLM running pipeline
    """
    global full_interaction

    # Load existing assets, tasks, and code
    base_tasks, base_assets, base_task_codes = get_base_memory(cfg['prompt_data_path'], cfg)
    start_time = time.time()
    prompt_folder = f"prompts/{cfg['prompt_folder']}"
    task_prompt_text = open(f"{prompt_folder}/cliport_prompt_task.txt").read()

    # Sample candidates from the base dictionary
    task_asset_replacement_str = format_dict_prompt(base_assets, cfg['task_asset_candidate_num'])
    task_descriptions_replacement_str = format_dict_prompt(base_tasks, cfg['task_description_candidate_num'])
    task_prompt_text = task_prompt_text.replace("TASK_NAME_PROMPT", task_descriptions_replacement_str)
    task_prompt_text = task_prompt_text.replace("TASK_ASSET_PROMPT", task_asset_replacement_str)
    res, full_interaction = generate_feedback(task_prompt_text, temperature=cfg['gpt_temperature'], interaction_txt=full_interaction)

    # Extract dictionary for task name, descriptions, and assets
    task_def = extract_dict(res, prefix='new_task')
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
        save_text(model_output_dir, f'{new_task["task-name"]}_asset_output', res)
        asset_list = extract_assets(res)
    else:
        asset_list = {}

    # API Preview
    if os.path.exists(f"{prompt_folder}/cliport_prompt_api_template.txt"):
        full_interaction = add_to_txt(full_interaction, "================= API Preview!", with_print=True)
        api_prompt_text = open(f'{prompt_folder}/cliport_prompt_api_template.txt').read()
        base_class_task_code = open(f'cliport/tasks/task.py').read()
        base_class_task_code = f'```\n{base_class_task_code}\n```\n\n'
        api_prompt_text = api_prompt_text.replace("TASK_CLASS_IMPLEMENTATION", base_class_task_code)
        api_prompt_text = api_prompt_text.replace("TASK_NAME_TEMPLATE", new_task["task-name"])
        res, full_interaction = generate_feedback(api_prompt_text, temperature=0, assistant_prompt=res, interaction_txt=full_interaction)
        # save_text(model_output_dir, f'{new_task["task-name"]}_api_output', res)

    # Error Preview
    if os.path.exists(f"{prompt_folder}/cliport_prompt_common_errors_template.txt"):
        full_interaction = add_to_txt(full_interaction, "================= Error Book Preview!", with_print=True)

        errorbook_prompt_text = open(f'{prompt_folder}/cliport_prompt_common_errors_template.txt').read()
        errorbook_prompt_text = errorbook_prompt_text.replace("TASK_NAME_TEMPLATE", new_task["task-name"])
        res, full_interaction = generate_feedback(errorbook_prompt_text, temperature=0., assistant_prompt=res, interaction_txt=full_interaction)
        # save_text(model_output_dir, f'{new_task["task-name"]}_error_summary_output', res)

    # Sample candidate code and compose
    if os.path.exists(f"{prompt_folder}/cliport_prompt_code_candidate_template.txt"):
        full_interaction = add_to_txt(full_interaction, "================= Code Generation!", with_print=True)

        if os.path.exists(f"{prompt_folder}/cliport_prompt_code_reference_selection_template.txt"):
            code_reference_question = open(f'{prompt_folder}/cliport_prompt_code_reference_selection_template.txt').read()
            code_reference_question = code_reference_question.replace("TASK_NAME_TEMPLATE", new_task["task-name"])
            code_reference_question = code_reference_question.replace("TASK_CODE_LIST_TEMPLATE", str(base_task_codes))
            code_reference_question = code_reference_question.replace("TASK_STRING_TEMPLATE", str(new_task))
            res, full_interaction = generate_feedback(code_reference_question, temperature=0., assistant_prompt=res, interaction_txt=full_interaction)
            code_reference_cmd = extract_list(res, prefix='code_reference')
            exec(code_reference_cmd, globals())
            task_code_reference_replace_prompt = sample_list_reference(code_reference, sample_num=-1)
        else:
            task_code_reference_replace_prompt = sample_list_reference(base_task_codes, sample_num=cfg['task_code_candidate_num'])

        code_prompt_text = open(f"{prompt_folder}/cliport_prompt_code_candidate_template.txt").read()
        code_prompt_text = code_prompt_text.replace("TASK_CODE_REFERENCE_TEMPLATE", task_code_reference_replace_prompt)
        code_prompt_text = code_prompt_text.replace("TASK_NAME_TEMPLATE", new_task["task-name"])
        code_prompt_text = code_prompt_text.replace("TASK_STRING_TEMPLATE", str(new_task))
        res, full_interaction = generate_feedback(code_prompt_text, temperature=0., assistant_prompt=res, interaction_txt=full_interaction) # cfg['gpt_temperature']

    code, task_name = extract_code(res)

    if len(task_name) == 0:
        print("empty task name:", task_name)
        return None

    # build the new environment class
    save_text(model_output_dir, task_name + '_code_output', code)
    try:
        exec(code, globals())
    except:
        print(str(traceback.format_exc()))
        return None

    cfg['task'] = new_task["task-name"]
    print(f"\n\nLLM generation time: {time.time() - start_time}")

    # save all output and return
    print("save all interaction to :", f'{new_task["task-name"]}_full_output')
    save_text(model_output_dir, f'{new_task["task-name"]}_full_output', full_interaction)
    return task_name, new_task, asset_list, code

@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    global full_interaction

    # Evaluation Metric
    SYNTAX_PASS_RATE = 0.
    RUNTIME_PASS_RATE = 0.
    ENV_PASS_RATE = 0.
    task_assets = []

    start_time = time.time()
    output_folder = 'output/output_stats'

    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    model_output_dir = os.path.join(output_folder, cfg['prompt_folder'] + "_" + model_time)
    TOTAL_TRIALS = cfg['trials']
    env_names = []

    for trial_i in range(TOTAL_TRIALS):
        # generate
        try:
            res = llm_gen_env(cfg, model_output_dir)
            if res is not None:
                SYNTAX_PASS_RATE += 1
                task_name, new_task, asset_list, code = res
                env_names.append(task_name)
                task_assets.append(new_task["assets-used"])
            else:
                env_names.append("")
                print("Syntax Failure")
                continue
        except:
            print("Syntax Failure")
            print(str(traceback.format_exc()))
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

            # Collect training data from oracle demonstrations.
            while dataset.n_episodes < cfg['n']:

                total_cnt += 1
                if total_cnt == cfg['max_env_run_cnt'] or total_cnt == cfg['n']:
                    # successfully resets and run
                    if reset_success_cnt == total_cnt - 1:
                        RUNTIME_PASS_RATE += 1
                        print("Runtime Test Pass!")

                        # the task can actually be completed with oracle
                        if env_success_cnt >= total_cnt / 2:
                            ENV_PASS_RATE += 1
                            print("Environment Test Pass!")

                        else:
                            # has success rates
                            print("Bad task design!")
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

                # Start video recording (NOTE: super slow)
                if record:
                    env.start_rec(f'{dataset.n_episodes+1:06d}')

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

                env_success_cnt += total_reward  > 0.99
                reset_success_cnt += 1

            p.disconnect()

        except:
            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
            save_text(model_output_dir, task_name + '_error', str(traceback.format_exc()))

            print("========================================================")
            print("Exception:", to_print)
            p.disconnect()

        if cfg['save_memory'] and env_success_cnt >= 1:
            # self-reflection
            save_base_memory(cfg['prompt_data_path'], new_task, asset_list, code, cfg)

        print("=========================================================")
        print(f"SYNTAX_PASS_RATE: {(SYNTAX_PASS_RATE / (trial_i+1)) * 100:.1f}% RUNTIME_PASS_RATE: {(RUNTIME_PASS_RATE / (trial_i+1)) * 100:.1f}% ENV_PASS_RATE: {(ENV_PASS_RATE / (trial_i+1)) * 100:.1f}%")
        print("=========================================================")

    DIVERSITY_SCORE = compute_diversity_score_from_assets(task_assets)
    save_stat(cfg, model_output_dir, env_names, SYNTAX_PASS_RATE / TOTAL_TRIALS, RUNTIME_PASS_RATE / TOTAL_TRIALS, ENV_PASS_RATE / TOTAL_TRIALS, DIVERSITY_SCORE)

if __name__ == '__main__':
    main()
