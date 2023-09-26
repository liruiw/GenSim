import numpy as np
import os
import IPython
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
from gensim.utils import (
    mkdir_if_missing,
    save_text,
    save_stat,
    compute_diversity_score_from_assets,
    add_to_txt,
    extract_dict,
    extract_code,
    extract_code_topdown,
    extract_code_topdown_offline
)
import pybullet as p
import copy
dummy_task = {"task-name": "dummy", "task-descriptions": "dummy", "assets-used": "dummy"}

class TopDownSimulationRunner:
    """ the main class that runs simulation loop """
    def __init__(self, cfg, memory):
        self.cfg = cfg
        self.memory = memory

        # statistics
        self.syntax_pass_rate = 0
        self.runtime_pass_rate = 0
        self.env_pass_rate = 0
        self.curr_trials = 0

        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.chat_log = memory.chat_log
        self.task_asset_logs = []

        # All the generated tasks in this run.
        # Different from the ones in online buffer that can load from offline.
        self.generated_task_assets = []
        self.generated_task_programs = []
        self.generated_task_names = []
        self.generated_tasks = []
        self.passed_tasks = [] # accepted ones

    def print_current_stats(self):
        """ print the current statistics of the simulation design """
        print("=========================================================")
        print(f"{self.cfg['prompt_folder']} Trial {self.curr_trials} SYNTAX_PASS_RATE: {(self.syntax_pass_rate / (self.curr_trials)) * 100:.1f}% RUNTIME_PASS_RATE: {(self.runtime_pass_rate / (self.curr_trials)) * 100:.1f}% ENV_PASS_RATE: {(self.env_pass_rate / (self.curr_trials)) * 100:.1f}%")
        print("=========================================================")

    def save_stats(self):
        """ save the final simulation statistics """
        self.diversity_score = compute_diversity_score_from_assets(self.task_asset_logs, self.curr_trials)
        save_stat(self.cfg, self.cfg['model_output_dir'], self.generated_tasks, self.syntax_pass_rate / (self.curr_trials),
                self.runtime_pass_rate / (self.curr_trials), self.env_pass_rate / (self.curr_trials), self.diversity_score)
        print("Model Folder: ", self.cfg['model_output_dir'])
        print(f"Total {len(self.generated_tasks)} New Tasks:", [task['task-name'] for task in self.generated_tasks])
        try:
            print(f"Added {len(self.passed_tasks)}  Tasks:", self.passed_tasks)
        except:
            pass


    def task_creation(self, res, offline_mode=False):
        """ create the task through interactions of agent and critic """
        self.task_creation_pass = True
        mkdir_if_missing(self.cfg['model_output_dir'])

        try:
            start_time = time.time()
            task_def = extract_dict(res, prefix="new_task")
            exec(task_def, globals())
            self.generated_task = new_task
        except:
            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
            print("Task Creation Exception:", to_print)
            self.generated_task = copy.deepcopy(dummy_task)
        try:
            self.generated_asset = {}
            extract_func = extract_code_topdown
            self.generated_code, self.curr_task_name = extract_func(res)

            self.task_asset_logs.append(self.generated_task["assets-used"])
            self.generated_task_name = self.generated_task["task-name"]
            self.generated_tasks.append(self.generated_task)
            self.generated_task_assets.append(self.generated_asset)
            self.generated_task_programs.append(self.generated_code)
            self.generated_task_names.append(self.generated_task_name)
        except:
            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
            print("Task Code Creation Exception:", to_print)
            self.task_creation_pass = False

        # self.curr_task_name = self.generated_task['task-name']
        print("task creation time {:.3f}".format(time.time() - start_time))


    def setup_env(self):
        """ build the new task"""
        env = Environment(
                self.cfg['assets_root'],
                disp=self.cfg['disp'],
                shared_memory=self.cfg['shared_memory'],
                hz=480,
                record_cfg=self.cfg['record']
            )

        task = eval(self.curr_task_name)()
        task.mode = self.cfg['mode']
        record = self.cfg['record']['save_video']
        save_data = self.cfg['save_data']

        # Initialize scripted oracle agent and dataset.
        expert = task.oracle(env)
        self.cfg['task'] = self.generated_task["task-name"]
        data_path = os.path.join(self.cfg['data_dir'], "{}-{}".format(self.generated_task["task-name"], task.mode))
        dataset = RavensDataset(data_path, self.cfg, n_demos=0, augment=False)
        print(f"Saving to: {data_path}")
        print(f"Mode: {task.mode}")

        # Start video recording
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        return task, dataset, env, expert

    def run_one_episode(self, dataset, expert, env, task, episode, seed):
        """ run the new task for one episode """
        add_to_txt(
                self.chat_log, f"================= TRIAL: {self.curr_trials}", with_print=True)
        record = self.cfg['record']['save_video']
        np.random.seed(seed)
        random.seed(seed)
        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, self.cfg['n'], seed))
        env.set_task(task)
        obs = env.reset()

        info = env.info
        reward = 0
        total_reward = 0

        # Rollout expert policy
        for _ in range(task.max_steps):
            act = expert.act(obs, info)
            episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break

        episode.append((obs, None, reward, info))
        return total_reward

    def simulate_task(self):
        """ simulate the created task and save demonstrations """
        total_cnt = 0.
        reset_success_cnt = 0.
        env_success_cnt = 0.
        seed = 123
        self.curr_trials += 1

        if p.isConnected():
            p.disconnect()

        if not self.task_creation_pass:
            print("task creation failure => count as syntax exceptions.")
            return

        # Check syntax and compilation-time error
        try:
            exec(self.generated_code, globals())
            task, dataset, env, expert = self.setup_env()
            self.syntax_pass_rate += 1

        except:
            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
            save_text(self.cfg['model_output_dir'], self.generated_task_name + '_error', str(traceback.format_exc()))
            print("========================================================")
            print("Syntax Exception:", to_print)
            return

        try:
            # Collect environment and collect data from oracle demonstrations.
            while total_cnt <= self.cfg['max_env_run_cnt']:
                total_cnt += 1
                episode = []
                total_reward = self.run_one_episode(dataset, expert, env, task, episode, seed)

                reset_success_cnt += 1
                env_success_cnt += total_reward > 0.99

            self.runtime_pass_rate += 1
            print("Runtime Test Pass!")

            # the task can actually be completed with oracle. 50% success rates are high enough.
            if env_success_cnt >= total_cnt / 2:
                self.env_pass_rate += 1
                print("Environment Test Pass!")

                if self.cfg['save_memory']:
                    self.memory.save_task_to_offline_topdown(self.generated_task, self.generated_code)
                    print(f"added new task to offline: {self.generated_task['task-name']}")

        except:
            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
            save_text(self.cfg['model_output_dir'], self.generated_task_name + '_error', str(traceback.format_exc()))
            print("========================================================")
            print("Runtime Exception:", to_print)
        self.memory.save_run(self.generated_task)


