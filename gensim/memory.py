import numpy as np
import os
import IPython

import random
import json
from gensim.utils import save_text


class Memory:
    """
    class that maintains a buffer of generated tasks and codes
    """
    def __init__(self, cfg):
        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.data_path = cfg["prompt_data_path"]
        self.cfg = cfg

        # a chat history is a list of strings
        self.chat_log = []
        self.online_task_buffer = {}
        self.online_code_buffer = {}
        self.online_asset_buffer = {}

        # directly load current offline memory into online memory
        base_tasks, base_assets, base_task_codes = self.load_offline_memory()
        self.online_task_buffer.update(base_tasks)
        self.online_asset_buffer.update(base_assets)

        # load each code file
        for task_file in base_task_codes:
            # the original cliport task path
            if os.path.exists("cliport/tasks/" + task_file):
                self.online_code_buffer[task_file] = open("cliport/tasks/" + task_file).read()

            # the generated cliport task path
            elif os.path.exists("cliport/generated_tasks/" + task_file):
                self.online_code_buffer[task_file] = open("cliport/generated_tasks/" + task_file).read()

        print(f"load {len(self.online_code_buffer)} tasks for memory from offline to online:")
        cache_embedding_path = "outputs/task_cache_embedding.npz"

        if os.path.exists(cache_embedding_path):
            print("task code embeding:", cache_embedding_path)
            self.task_code_embedding = np.load(cache_embedding_path)

    def save_run(self, new_task):
        """save chat history and potentially save base memory"""
        print("save all interaction to :", f'{new_task["task-name"]}_full_output')
        unroll_chatlog = ''
        for chat in self.chat_log:
            unroll_chatlog += chat
        save_text(
            self.cfg['model_output_dir'], f'{new_task["task-name"]}_full_output', unroll_chatlog
        )

    def save_task_to_online(self, new_task, code):
        """(not dumping the task offline). save the task information for online bootstrapping."""
        self.online_task_buffer[new_task['task-name']] = new_task
        code_file_name = new_task["task-name"].replace("-", "_") + ".py"

        # code file name: actual code in contrast to offline code files format.
        self.online_code_buffer[code_file_name] = code

    def save_task_to_offline(self, new_task, code, generate_task_path='generated_tasks'):
        """save the current task descriptions, assets, and code, if it passes reflection and environment test"""
        generated_task_code_path = os.path.join(
            self.cfg["prompt_data_path"], f"generated_task_codes.json"
        )
        generated_task_codes = json.load(open(generated_task_code_path))
        new_file_path = new_task["task-name"].replace("-", "_") + ".py"

        if new_file_path not in generated_task_codes:
            generated_task_codes.append(new_file_path)

            python_file_path = f"cliport/{generate_task_path}/{new_file_path}"
            print(f"save {new_task['task-name']} to ", python_file_path)

            with open(python_file_path, "w") as fhandle:
                fhandle.write(code)

            with open(generated_task_code_path, "w") as outfile:
                json.dump(generated_task_codes, outfile, indent=4)
        else:
            print(f"{new_file_path}.py already exists.")

        # save task descriptions
        generated_task_path = os.path.join(
           self.cfg["prompt_data_path"], f"{generate_task_path}.json"
        )
        generated_tasks = json.load(open(generated_task_path))
        generated_tasks[new_task["task-name"]] = new_task

        with open(generated_task_path, "w") as outfile:
            json.dump(generated_tasks, outfile, indent=4)

    def save_task_to_offline_topdown(self, new_task, code, generate_task_path='topdown_generated_tasks'):
        new_file_path = new_task["task-name"].replace("-", "_") + ".py"
        generated_task_codes.append(new_file_path)

        python_file_path = f"cliport/{generate_task_path}/{new_file_path}"
        print(f"save {new_task['task-name']} to ", python_file_path)

        with open(python_file_path, "w") as fhandle:
            fhandle.write(code)


    def load_offline_memory(self):
        """get the current task descriptions, assets, and code"""
        base_task_path = os.path.join(self.data_path, "base_tasks.json")
        base_asset_path = os.path.join(self.data_path, "base_assets.json")
        base_task_code_path = os.path.join(self.data_path, "base_task_codes.json")

        base_tasks = json.load(open(base_task_path))
        base_assets = json.load(open(base_asset_path))
        base_task_codes = json.load(open(base_task_code_path))

        if self.cfg["load_memory"]:
            generated_task_path = os.path.join(self.data_path, "generated_tasks.json")
            generated_asset_path = os.path.join(self.data_path, "generated_assets.json")
            generated_task_code_path = os.path.join(self.data_path, "generated_task_codes.json")

            print("original base task num:", len(base_tasks))
            base_tasks.update(json.load(open(generated_task_path)))
            # base_assets.update(json.load(open(generated_asset_path)))

            for task in json.load(open(generated_task_code_path)):
                if task not in base_task_codes:
                    base_task_codes.append(task)

            print("current base task num:", len(base_tasks))
        return base_tasks, base_assets, base_task_codes
