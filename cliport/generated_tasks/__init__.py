import os
from pprint import pprint

# automatically import all defined task classes in this directory
new_names = {}
dir_path = os.path.dirname(os.path.realpath(__file__))
for file in os.listdir(dir_path):
    if 'init' not in file and 'cache' not in file:
        code_file = open(f"{dir_path}/{file}").read()
        code_lines = code_file.split("\n")
        class_def = [line for line in code_lines if line.startswith('class')]
        task_name = class_def[0]
        task_name = task_name[task_name.find("class "): task_name.rfind("(Task)")][6:]
        file_name = file.replace('.py','')
        exec(f"from cliport.generated_tasks.{file_name} import {task_name}")
        new_names[file_name.replace("_", "-")] = eval(task_name)


# pprint(new_names)
