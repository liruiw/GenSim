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

class ColorSortedBlocksOnPalletWithBarrier(Task):
    """Pick up colored blocks, navigate over a barrier, and place them on a pallet in a specific color sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} block on the pallet"
        self.task_completed_desc = "done placing blocks on the pallet."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.15, 0.15, 0.02)
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object('pallet/pallet.urdf', pallet_pose, 'fixed')

        # Add barrier.
        barrier_size = (0.02, 0.02, 0.02)
        barrier_pose = self.get_random_pose(env, barrier_size)
        env.add_object('block/small.urdf', barrier_pose, 'fixed')

        # Block colors.
        colors = ['red', 'blue', 'green', 'yellow']
        color_names = ['red', 'blue', 'green', 'yellow']

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[colors[i]])
            blocks.append(block_id)

        # Goal: each block is on the pallet in the specific color sequence.
        for i in range(4):
            language_goal = self.lang_template.format(color=color_names[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[pallet_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 4,
                          language_goal=language_goal)