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

class MultiLevelPyramidConstruction(Task):
    """Construct a two-level pyramid on a pallet using six blocks: three green and three blue."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Construct a two-level pyramid on a pallet using six blocks: three green and three blue. The first level should be a triangle created by placing the green blocks side by side. The second level should be built by placing the blue blocks on top of the green blocks, forming another triangle rotated 60 degrees with respect to the first one."
        self.task_completed_desc = "done constructing pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.35, 0.35, 0.01)  # x, y, z dimensions for the pallet size
        pallet_pose = self.get_random_pose(env, pallet_size)
        pallet_urdf = 'pallet/pallet.urdf'
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)  # x, y, z dimensions for the block size
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['green']] * 3 + [utils.COLORS['blue']] * 3  # three green and three blue blocks

        blocks = []
        for color in block_colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=color)
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.02), (0, 0, 0.02), (0, 0.05, 0.02),  # first level
                     (0, -0.025, 0.06), (0, 0.025, 0.06), (0, 0, 0.10)]  # second level
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid (first level: green blocks).
        self.add_goal(objs=blocks[:3], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3,
                          language_goal=self.lang_template.format(blocks="the green blocks", row="bottom"))

        # Goal: blocks are stacked in a pyramid (second level: blue blocks).
        self.add_goal(objs=blocks[3:], matches=np.ones((3, 3)), targ_poses=targs[3:], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3,
                          language_goal=self.lang_template.format(blocks="the blue blocks", row="top"))