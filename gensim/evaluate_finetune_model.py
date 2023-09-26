import openai
import argparse
import os
from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

import time
import random
import json
import traceback
import pybullet as p
import IPython
from gensim.topdown_sim_runner import TopDownSimulationRunner
import hydra
from datetime import datetime

from gensim.memory import Memory
from gensim.utils import set_gpt_model, clear_messages, format_finetune_prompt

@hydra.main(config_path='../cliport/cfg', config_name='data', version_base="1.2")
def main(cfg):
    task = cfg.target_task
    model = cfg.target_model
    prompt = format_finetune_prompt(task)

    openai.api_key = cfg['openai_key']
    # model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

    #
    cfg['model_output_dir'] = os.path.join(cfg['output_folder'], cfg['prompt_folder'] + "_" + cfg.target_model)
    if 'seed' in cfg:
       cfg['model_output_dir'] = cfg['model_output_dir'] + f"_{cfg['seed']}"

    set_gpt_model(cfg['gpt_model'])
    memory = Memory(cfg)
    simulation_runner = TopDownSimulationRunner(cfg, memory)

    for trial_i in range(cfg['trials']):
        if 'new_finetuned_model' in cfg or 'gpt-3.5-turbo' in cfg.target_model:
                # the chat completion version
                response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You are an AI in robot simulation code and task design."},
                          {"role": "user", "content": prompt}],
                temperature=0.01,
                max_tokens=1000,
                n=1,
                stop=["\n```\n"])
                res = response["choices"][0]["message"]["content"]
        else:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=0,
                max_tokens=1800,
                stop=["\n```\n"])
            res = response["choices"][0]["text"]

        simulation_runner.task_creation(res)
        simulation_runner.simulate_task()
        simulation_runner.print_current_stats()

    simulation_runner.save_stats()




# load few shot prompts


if __name__ == "__main__":
    main()
