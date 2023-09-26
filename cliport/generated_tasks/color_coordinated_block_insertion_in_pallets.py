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

class ColorCoordinatedBlockInsertion(Task):
    """Pick up each colored block and place it into the pallet of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {} block into the {} pallet"
        self.task_completed_desc = "done placing blocks into pallets."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
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

        # Add pallets.
        pallet_size = (0.12, 0.12, 0)
        pallet_urdf = 'pallet/pallet.urdf'
        pallets = []
        for i in range(4):
            pallet_pose = self.get_random_pose(env, pallet_size)
            pallet_id = env.add_object(pallet_urdf, pallet_pose, color=utils.COLORS[colors[i]], category='fixed')
            pallets.append(pallet_id)

        # Goal: each block is in the pallet of the same color.
        for i in range(4):
            language_goal = self.lang_template.format(color_names[i], color_names[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[pallet_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4,
                          language_goal=language_goal)