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

class SortingBlocksIntoPallets(Task):
    """Pick up blocks of four different colors (red, blue, green, yellow) and place them into four separate pallets of matching color. The pallets are placed in a row and the blocks are scattered randomly on the table."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {color} block into the {color} pallet"
        self.task_completed_desc = "done sorting blocks into pallets."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallets.
        # x, y, z dimensions for the asset size
        pallet_size = (0.12, 0.12, 0.02)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_poses = []
        pallet_colors = ['red', 'blue', 'green', 'yellow']
        for color in pallet_colors:
            pallet_pose = self.get_random_pose(env, pallet_size)
            env.add_object(pallet_urdf, pallet_pose, 'fixed', color=utils.COLORS[color])
            pallet_poses.append(pallet_pose)

        # Add blocks.
        # x, y, z dimensions for the asset size
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        for color in pallet_colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Goal: each block is in a different pallet of matching color.
        for i in range(len(blocks)):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[pallet_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/len(blocks), 
                language_goal=self.lang_template.format(color=pallet_colors[i]))