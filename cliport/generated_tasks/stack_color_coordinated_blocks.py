import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class StackColorCoordinatedBlocks(Task):
    """Pick up six blocks of different colors (red, blue, green, yellow, orange, and purple) 
    and stack them on a pallet in two separate stacks. The first stack should be red at the bottom, 
    blue in the middle, and green at top. The second stack should be yellow at the bottom, 
    orange in the middle, and purple at the top."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "stack the blocks on the pallet in two separate stacks. " \
                             "The first stack should be red at the bottom, blue in the middle, " \
                             "and green at top. The second stack should be yellow at the bottom, " \
                             "orange in the middle, and purple at the top."
        self.task_completed_desc = "done stacking color-coordinated blocks."

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        # x, y, z dimensions for the asset size
        pallet_size = (0.15, 0.15, 0.01)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['orange'], utils.COLORS['purple']
        ]

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'box/box-template.urdf'
        blocks = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.02), (0, 0, 0.02), (0, 0.05, 0.02), 
                     (0, -0.05, 0.06), (0, 0, 0.06), (0, 0.05, 0.06)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: blocks are stacked on the pallet in two separate stacks.
        # First stack: red at the bottom, blue in the middle, and green at top.
        self.add_goal(objs=blocks[:3], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3,
                language_goal=self.lang_template)

        # Second stack: yellow at the bottom, orange in the middle, and purple at the top.
        self.add_goal(objs=blocks[3:], matches=np.ones((3, 3)), targ_poses=targs[3:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3,
                language_goal=self.lang_template)