import re
import os

import os
import json
import argparse

import IPython

def extract_dict(res, task_name, prefix="new_task"):
    """ parse task dictionary from the code itself """
    task_dict =  {"task-name": task_name,
    "assets-used": []}
    pattern = r'\'(.*?).urdf'
    asset_string = re.findall(pattern, res)

    pattern = r'"""(.*?)"""'
    description_string = re.findall(pattern, res, re.DOTALL)
    task_dict["assets-used"] = [file + ".urdf" for file in asset_string]
    task_dict["task-description"] = description_string[0]
    print(description_string[0])
    print(asset_string)
    return task_dict


# remove some tasks from the list
parser = argparse.ArgumentParser()

parser.add_argument(
    "--files", "-f", type=str, default="exps"
)
args = parser.parse_args()


data_path = "prompts/data"
generated_task_path = os.path.join(data_path, "generated_tasks.json")
generated_task_code_path = os.path.join(data_path, "generated_task_codes.json")

generated_tasks = json.load(open(generated_task_path))
generated_task_codes = json.load(open(generated_task_code_path))


task_names = args.files.split(",")
print("Task names:", task_names)

for task_name in task_names:

  task_name = task_name.replace("_", "-")
  task_name_py = task_name.replace("-", "_") + ".py"
  file_path = "cliport/generated_tasks/" + task_name_py
  if os.path.exists(file_path) and task_name not in generated_tasks:
    print("add task:", task_name)

    code = open(file_path).read()
    generated_tasks[task_name] = extract_dict(code, task_name)

    if task_name_py not in generated_task_codes:
      generated_task_codes.append(task_name_py)

with open(generated_task_code_path, "w") as outfile:
    json.dump(generated_task_codes, outfile, indent=4)

with open(generated_task_path, "w") as outfile:
    json.dump(generated_tasks, outfile, indent=4)

