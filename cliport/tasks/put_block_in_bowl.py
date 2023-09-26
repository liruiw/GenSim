import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p


class PutBlockInBowl(Task):
    """Place all blocks of a specified color in a bowl of specified color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "put the {pick} blocks in a {place} bowl"
        self.task_completed_desc = "done placing blocks in bowls."
        self.additional_reset()


    def reset(self, env):
        super().reset(env)
        n_bowls = np.random.randint(1, 4)
        n_blocks = np.random.randint(1, n_bowls + 1)
        colors, selected_color_names = utils.get_colors(mode=self.mode, n_colors=2)

        # Add bowls.
        # x, y, z dimensions for the asset size
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, category='fixed', color=colors[1])
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[0])
            blocks.append(block_id)

        # Goal: put each block in a different bowl.
        language_goal = (self.lang_template.format(pick=selected_color_names[0], place=selected_color_names[1]))
        self.add_goal(objs=blocks, matches=np.ones((len(blocks), len(bowl_poses))), targ_poses=bowl_poses, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1, language_goal=language_goal)
        

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

        # Colors of distractor objects.
        distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        while n_distractors < max_distractors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_block_colors if is_block else distractor_bowl_colors
            pose = self.get_random_pose(env, size)
            color = colors[n_distractors % len(colors)]

            obj_id = env.add_object(urdf, pose, color=color)
            n_distractors += 1