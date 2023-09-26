import numpy as np
import os
import IPython

import traceback
import json
from gensim.utils import (
    save_text,
    add_to_txt,
    extract_dict,
    format_dict_prompt,
    generate_feedback,
)
import copy
import random

class Critic:
    """
    class that reflects and criticizes new task for improvement
    """
    def __init__(self, cfg, memory):
        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.memory = memory
        self.chat_log = self.memory.chat_log
        self.cfg = cfg
        self.model_output_dir = cfg["model_output_dir"]

    def error_review(self, new_task):
        """ commonly made error review """
        if os.path.exists(f"{self.prompt_folder}/cliport_prompt_common_errors_template.txt") and "task-name" in new_task:
            self.chat_log = add_to_txt(self.chat_log, "================= Error Book Preview!", with_print=True)
            errorbook_prompt_text = open(f'{self.prompt_folder}/cliport_prompt_common_errors_template.txt').read()
            errorbook_prompt_text = errorbook_prompt_text.replace("TASK_NAME_TEMPLATE", new_task["task-name"])
            res = generate_feedback(errorbook_prompt_text, temperature=0., interaction_txt=self.chat_log) # cfg['gpt_temperature']

    def reflection(self, new_task, new_code, current_tasks=None):
        """ reflect on if the new task needs to be added """
        all_add_to_the_task_list_flag = True

        if os.path.exists(f"{self.prompt_folder}/cliport_prompt_task_reflection.txt"):
            # only consider successful task
            self.chat_log = add_to_txt(self.chat_log, "================= Code Reflect!", with_print=True)
            total_tasks = copy.deepcopy(self.memory.online_task_buffer)
            if current_tasks is not None:
                # adding all the tasks in the current run. at least should not overlap with those
                for t in current_tasks:
                    total_tasks[t['task-name']] = t

            # need to load more
            total_tasks = self.memory.online_task_buffer
            MAX_NUM = 20
            if len(total_tasks) > MAX_NUM:
                total_tasks = dict(random.sample(total_tasks.items(), MAX_NUM))

            print("reflection history task num:", len(total_tasks))
            task_descriptions_replacement_str = format_dict_prompt(total_tasks, -1)

            # append current new task
            code_reflection_prompt_text = open(f"{self.prompt_folder}/cliport_prompt_task_reflection.txt").read()
            code_reflection_prompt_text = code_reflection_prompt_text.replace("CURRENT_TASK_NAME_TEMPLATE", str(task_descriptions_replacement_str))
            code_reflection_prompt_text = code_reflection_prompt_text.replace("TASK_STRING_TEMPLATE", str(new_task))
            code_reflection_prompt_text = code_reflection_prompt_text.replace("TASK_CODE_TEMPLATE", str(new_code))
            if len(self.cfg['target_task_name']) > 0:
                code_reflection_prompt_text = code_reflection_prompt_text.replace("TARGET_TASK_NAME", self.cfg['target_task_name'])

            # no matter
            total_tasks[new_task["task-name"].replace("-", "_")] = str(new_task)
            res = generate_feedback(code_reflection_prompt_text, temperature=0.4, interaction_txt=self.chat_log, n=int(self.cfg['reflection_agreement_num'])) # cfg['gpt_temperature']
            all_add_to_the_task_list_flag = True

            for idx, r in enumerate(res):
                # iterate through for agreement
                reflection_def_cmd = extract_dict(r, prefix='task_reflection')
                exec(reflection_def_cmd, globals())
                try:
                    print(f"critic {idx}:", task_reflection)

                    if task_reflection["add_to_the_task_list"] == 'False':
                        all_add_to_the_task_list_flag = False
                        print(f"critic {idx} suggests not adding this task to the buffer! ")
                except:
                    print("bug")
                    pass

            save_text(self.model_output_dir, new_task['task-name'] + "_reflection_output", str(task_reflection))

        return all_add_to_the_task_list_flag