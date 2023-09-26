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

class ColorMatchingBlockInPallet(Task):
    """Arrange four different colored blocks (red, blue, green, yellow) on a pallet. 
    The blocks should be placed such that they form a square, with the red block in the top left, 
    blue block in the top right, green block in the bottom left, and yellow block in the bottom right."""

    def __init__(self):
        super().__init__()
        self.max_steps = 4
        self.lang_template = "place the {color} block at the {position} of the pallet"
        self.task_completed_desc = "done arranging blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.35, 0.35, 0.01)  # x, y, z dimensions for the pallet size
        pallet_pose = self.get_random_pose(env, pallet_size)
        pallet_urdf = 'pallet/pallet.urdf'
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Define block colors and corresponding positions.
        colors = ['red', 'blue', 'green', 'yellow']
        positions = ['top left', 'top right', 'bottom left', 'bottom right']

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)  # x, y, z dimensions for the block size
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[colors[i]])
            blocks.append(block_id)

            # Add goal for each block.
            dx = -0.1 if 'left' in positions[i] else 0.1
            dy = -0.1 if 'bottom' in positions[i] else 0.1
            dz = block_size[2] / 2
            target_pose = utils.apply(pallet_pose, (dx, dy, dz))

            language_goal = self.lang_template.format(color=colors[i], position=positions[i])
            self.add_goal(objs=[block_id], matches=np.ones((1, 1)), targ_poses=[target_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 4, language_goal=language_goal)