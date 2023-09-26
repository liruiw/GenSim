import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import pybullet as p
import random

class StackBlockPyramidSeq(Task):
    """Stacking Block Pyramid Sequence base class."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "put the {pick} block on {place}"
        self.task_completed_desc = "done stacking block pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add base.
        # x, y, z dimensions for the asset size
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        colors, color_names = utils.get_colors(self.mode)

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'

        objs = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            objs.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: make bottom row.
        language_goal = (self.lang_template.format(pick=color_names[0], place="the lightest brown block"))        
        self.add_goal(objs=[objs[0]], matches=np.ones((1, 1)), targ_poses=[targs[0]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2], language_goal=language_goal)

        language_goal = (self.lang_template.format(pick=color_names[1], place="the middle brown block"))
        self.add_goal(objs=[objs[1]], matches=np.ones((1, 1)), targ_poses=[targs[1]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2], language_goal=language_goal)

        language_goal = (self.lang_template.format(pick=color_names[2], place="the darkest brown block"))
        self.add_goal(objs=[objs[2]], matches=np.ones((1, 1)), targ_poses=[targs[2]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2], language_goal=language_goal)

        # Goal: make middle row.
        language_goal = (self.lang_template.format(pick=color_names[3], place=f"the {color_names[0]} and {color_names[1]} blocks"))        
        self.add_goal(objs=[objs[3]], matches=np.ones((1, 1)), targ_poses=[targs[3]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2], language_goal=language_goal)

        language_goal = (self.lang_template.format(pick=color_names[4], place=f"the {color_names[1]} and {color_names[2]} blocks"))
        self.add_goal(objs=[objs[4]], matches=np.ones((1, 1)), targ_poses=[targs[4]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2], language_goal=language_goal)

        # Goal: make top row.
        language_goal = (self.lang_template.format(pick=color_names[5], place=f"the {color_names[3]} and {color_names[4]} blocks"))
        self.add_goal(objs=[objs[5]], matches=np.ones((1, 1)), targ_poses=[targs[5]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2], language_goal=language_goal)