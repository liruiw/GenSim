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

class MultiLevelBlockConstruction(Task):
    """Construct a two-level structure on a pallet using four blocks: two red and two blue. 
    The lower level should be a rectangle created by placing the red blocks side by side. 
    The upper level is made up by placing the blue blocks placed on top of the red blocks 
    creating a line aligned perpendicular to the red blocks."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "construct a two-level structure on a pallet using four blocks: two red and two blue"
        self.task_completed_desc = "done constructing multi-level block structure."

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
        block_colors = [utils.COLORS['red'], utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['blue']]
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.02, 0.02), (0, 0.02, 0.02),
                     (0, -0.02, 0.06), (0, 0.02, 0.06)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: red blocks are placed side by side on the pallet.
        self.add_goal(objs=blocks[:2], matches=np.ones((2, 2)), targ_poses=targs[:2], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*2, language_goal=self.lang_template)

        # Goal: blue blocks are stacked on top of the red blocks.
        self.add_goal(objs=blocks[2:], matches=np.ones((2, 2)), targ_poses=targs[2:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*2, language_goal=self.lang_template)