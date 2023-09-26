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

class BlockInsertionInCornerPyramid(Task):
    """Insert each block into the corresponding color coded slot in the pyramid from bottom (red) to top (yellow)."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert the {color} block into the corresponding slot in the pyramid"
        self.task_completed_desc = "done inserting blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pyramid.
        pyramid_size = (0.15, 0.15, 0.15)
        pyramid_pose = self.get_random_pose(env, pyramid_size)
        pyramid_urdf = 'corner/corner-template.urdf'
        env.add_object(pyramid_urdf, pyramid_pose, 'fixed')

        # Block colors.
        colors = ['red', 'blue', 'green', 'yellow']

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for color in colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Goal: each block is inserted into the corresponding color coded slot in the pyramid.
        for i, block in enumerate(blocks):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[block], matches=np.ones((1, 1)), targ_poses=[pyramid_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 4, language_goal=language_goal)