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

from gensim.utils import set_gpt_model, clear_messages, format_finetune_prompt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='build-car')
    parser.add_argument("--model", type=str, default='davinci:ft-wang-lab:gensim-2023-08-05-16-54-05')
    # davinci:ft-mit-cal:gensim-2023-08-06-16-00-56
    args = parser.parse_args()
    task = args.task
    prompt = format_finetune_prompt(task)

    if True:
        response = openai.Completion.create(
            model=args.model,
            prompt=prompt,
            temperature=0,
            max_tokens=1024)
        res = response["choices"][0]["text"]
    else:
        params = {
            "model": args.model,
            "max_tokens": 500,
            "temperature": 0.1,
            "messages": [prompt]
        }
        call_res = openai.ChatCompletion.create(**params)
        res = call_res["choices"][0]["message"]["content"]

    print("code!:", res)
    python_file_path = f"cliport/generated_tasks/finetune_{task.replace('-','_')}.py"
    print(f"saving task {args.task} to {python_file_path}")

    # evaluate and then save
    # with open(python_file_path, "w",
        #         ) as fhandle:
        # fhandle.write(res)

