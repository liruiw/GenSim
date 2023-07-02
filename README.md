# Supersizing Simulation Task Generation in Robotics with LLM

This repo explores the use of an agent-based LLM code generation pipeline to write simulation environments and expert goals to augment diverse simulation tasks. This simulation task generation pipeline can be top-down: given a target task, it proposes a task curriculum to iteratively approach the complexity of the target task; the pipeline can also be bottom-up: it bootstraps on previous tasks and iteratively proposes more interesting tasks. Since the task is defined by simulation code, we can also train a generalist policy on top of the generated environments and tasks. See `BLOG.md` for a full discussion.
![](media/generated_task.gif)

## Installation
0. ``pip install -r requirements.txt``
1. ``python setup.py develop``
2. ``export AUTOSIM_ROOT=$(pwd)``
3. ``export OPENAI_KEY=YOUR KEY``. We use OpenAI's GPT-4 as the language model. You need to have an OpenAI API key to run task generation with LLMSim. You can get one from [here](https://platform.openai.com/account/api-keys).


## Getting Started
After the installation process, you can run: 
```
# basic bottom-up prompt
python autosim/run_simulation.py  disp=True   prompt_folder=vanilla_task_generation_prompt trials=5

# bottom-up template generation
python autosim/run_simulation.py    disp=True prompt_folder=bottomup_task_generation_prompt trials=10   save_memory=True load_memory=True  task_description_candidate_num=10 use_template=True

# top-down task generation
python autosim/run_simulation.py  disp=True  prompt_folder=topdown_task_generation_prompt trials=5  save_memory=True load_memory=True task_description_candidate_num=10 use_template=True target_task_name="build-house"

# task-conditioned chain-of-thought generation
python autosim/run_simulation.py  disp=True  prompt_folder=topdown_chain_of_thought_prompt trials=5 save_memory=True load_memory=True task_description_candidate_num=10 use_template=True target_task_name="build-car"  
```


## LLM Generated Tasks
1. All generated tasks in `cliport/generated_tasks` should have automatically been imported
2. Just change the name to the corresponding classes and then use `demo.py`. For instance, `python cliport/demos.py n=200 task=build-car mode=test disp=True`.
3.  The following is a guide for training everything from scratch (More details in [cliport](https://github.com/cliport/cliport)). All tasks follow a 4-phase workflow:
    1. Generate `train`, `val`, `test` datasets with `demos.py` 
    2. Train agents with `train.py` 
    3. Run validation with `eval.py` to find the best checkpoint on `val` tasks and save `*val-results.json`
    4. Evaluate the best checkpoint in `*val-results.json` on `test` tasks with `eval.py`


## Note
0. Temperature 0.5-0.8 is good range for diversity, 0.0-0.2 is for stable results.
1. The generation pipeline will print out statistics regarding compilation, runtime, task design, and diversity scores.
2. Core prompting and code generation scripts are in `autosim` and training and task scripts are in `cliport`.
3. `prompts/` folder stores different kinds of prompts to get the desired environments. Each folder contains a sequence of prompts as well as a meta_data file. `prompts/data` stores the base task library and the generated task library.
4. The GPT-generated tasks are stored in `generated_tasks/`. Use `demo.py` to play with them.  `cliport/demos_gpt4.py` deprecated all-in-one prompt script.
5. Raw text outputs are saved in `output/output_stats`, figure results saved in `output/output_figures`, policy evaluation results are saved in `output/cliport_output`.
6. To debug generated code, manually copy-paste ``generated_task.py`` then run 
``python cliport/demos.py n=50 task=gen-task disp=True``
7. This version of cliport should support `batchsize>1` and can run with more recent versions of pytorch and pytorch lightning.
8. Please use Github issue tracker to report bugs. For other questions please contact [Lirui Wang](wangliruisz@gmail.com)