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

class PyramidBlocksAssemble(Task):
    """Construct a pyramid using nine blocks in a specific color order on a pallet."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "construct a pyramid using nine blocks in a specific color order on a pallet"
        self.task_completed_desc = "done constructing pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        # x, y, z dimensions for the asset size
        pallet_size = (0.35, 0.35, 0.01)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['orange']
        ]

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for _ in range(9):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [
            (-0.1, -0.1, 0.02), (0, -0.1, 0.02), (0.1, -0.1, 0.02), (-0.1, 0, 0.02), (0.1, 0, 0.02),
            (-0.05, 0.05, 0.06), (0.05, 0.05, 0.06), (0, 0.1, 0.06),
            (0, 0.05, 0.1)
        ]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid in a specific color order.
        for i in range(9):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 9, 
                          language_goal=self.lang_template.format(blocks="the blocks",
                                                             row="row"))