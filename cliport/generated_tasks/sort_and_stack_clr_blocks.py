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

class SortAndStackClrBlocks(Task):
    """Pick up four blocks of different colors (red, blue, green, yellow) and place them into separate corners of a pallet. After sorting, stack them in a specific sequence on top of the pallet. The bottom of the stack should start with a green block followed by a blue, then red, and finally a yellow block at the top."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "sort and stack the blocks in the order of green, blue, red, and yellow"
        self.task_completed_desc = "done sorting and stacking blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        # x, y, z dimensions for the asset size
        pallet_size = (0.15, 0.15, 0.01)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Block colors.
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']]

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0.05, 0.05, 0.02), (-0.05, 0.05, 0.02), (-0.05, -0.05, 0.02), (0.05, -0.05, 0.02)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: blocks are sorted into separate corners of the pallet.
        self.add_goal(objs=blocks, matches=np.eye(4), targ_poses=targs, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=0.5, symmetries=[np.pi/2]*4,
                          language_goal=self.lang_template)

        # Associate stacking locations for goals.
        stack_pos = [(0, 0, 0.02), (0, 0, 0.06), (0, 0, 0.10), (0, 0, 0.14)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in stack_pos]

        # Goal: blocks are stacked on top of the pallet in the order of green, blue, red, and yellow.
        self.add_goal(objs=blocks, matches=np.eye(4), targ_poses=targs, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=0.5, symmetries=[np.pi/2]*4,
                          language_goal=self.lang_template)