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

class ColorCoordinatedBlockTower(Task):
    """Stack four blocks on a pallet in the following order from bottom to top: 
    two blue blocks side by side, one red block centered on the blue blocks, 
    and one green block on top of the red block."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack four blocks on a pallet in the following order from bottom to top: two blue blocks side by side, one red block centered on the blue blocks, and one green block on top of the red block."
        self.task_completed_desc = "done stacking blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        # x, y, z dimensions for the asset size
        pallet_size = (0.15, 0.15, 0.015)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['blue'], utils.COLORS['blue'], utils.COLORS['red'], utils.COLORS['green']]
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.02, 0.02), (0, 0.02, 0.02), (0, 0, 0.06), (0, 0, 0.10)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: two blue blocks are placed side by side on the pallet.
        # Break the language prompt step-by-step
        self.add_goal(objs=blocks[:2], matches=np.ones((2, 2)), targ_poses=targs[:2], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*2,
                language_goal="place two blue blocks side by side on the pallet")

        # Goal: one red block is placed centered on the blue blocks.
        self.add_goal(objs=blocks[2:3], matches=np.ones((1, 1)), targ_poses=targs[2:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2],
                language_goal="place one red block centered on the blue blocks")

        # Goal: one green block is placed on top of the red block.
        self.add_goal(objs=blocks[3:], matches=np.ones((1, 1)), targ_poses=targs[3:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2],
                language_goal="place one green block on top of the red block")