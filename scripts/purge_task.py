import os
import json
import argparse

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
	print("purge task:", task_name)
	task_name_py = task_name.replace("-", "_") + ".py"
	del generated_tasks[task_name]
	generated_task_codes.remove(task_name_py)
	os.system("rm cliport/generated_tasks/" + task_name_py)

with open(generated_task_code_path, "w") as outfile:
    json.dump(generated_task_codes, outfile, indent=4)

with open(generated_task_path, "w") as outfile:
    json.dump(generated_tasks, outfile, indent=4)