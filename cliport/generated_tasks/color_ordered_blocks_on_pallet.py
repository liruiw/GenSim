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

class ColorOrderedBlocksOnPallet(Task):
    """Pick up each colored block and place it onto the pallet in specific color sequence: red, blue, green, yellow, orange, and finally purple."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the colored blocks onto the pallet in the following order: red, blue, green, yellow, orange, and purple"
        self.task_completed_desc = "done placing blocks on the pallet."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        # x, y, z dimensions for the asset size
        pallet_size = (0.15, 0.15, 0.02)
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
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: blocks are placed on the pallet in the order of red, blue, green, yellow, orange, purple.
        for i in range(6):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2], language_goal=self.lang_template)