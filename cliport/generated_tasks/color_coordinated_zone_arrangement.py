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
import pybullet as p

class ColorCoordinatedZoneArrangement(Task):
    """Pick up blocks of different colors and place them on the pallets of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "place the {color} blocks on the {color} pallet"
        self.task_completed_desc = "done arranging blocks on pallets."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallets.
        # x, y, z dimensions for the asset size
        pallet_size = (0.12, 0.12, 0.02)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_colors = ['red', 'blue', 'green']
        pallet_poses = []
        for color in pallet_colors:
            pallet_pose = self.get_random_pose(env, pallet_size)
            env.add_object(pallet_urdf, pallet_pose, category='fixed', color=utils.COLORS[color])
            pallet_poses.append(pallet_pose)

        # Add blocks.
        # x, y, z dimensions for the asset size
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        for color in pallet_colors:
            for _ in range(3):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
                blocks.append(block_id)

        # Add small blocks as obstacles.
        small_block_size = (0.02, 0.02, 0.02)
        small_block_urdf = 'block/small.urdf'
        for _ in range(5):
            small_block_pose = self.get_random_pose(env, small_block_size)
            env.add_object(small_block_urdf, small_block_pose)

        # Goal: each block is on the pallet of the same color.
        for i in range(9):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[pallet_poses[i // 3]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 9,
                          language_goal=self.lang_template.format(color=pallet_colors[i // 3]))