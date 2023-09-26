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

class ArrangeBlocksInLine(Task):
    """Arrange four differently colored blocks (red, blue, green, yellow) in a line on the tabletop in the order of the colors of the rainbow."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "arrange the blocks in a line in the order of the colors of the rainbow"
        self.task_completed_desc = "done arranging blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # IMPORTANT Associate placement locations for goals.
        place_pos = [(0, -0.05, 0), (0, 0, 0), (0, 0.05, 0), (0, 0.1, 0)]
        targs = [(utils.apply(block_pose, i), block_pose[1]) for i in place_pos]

        # Goal: blocks are arranged in a line in the order of the colors of the rainbow.
        self.add_goal(objs=blocks, matches=np.ones((4, 4)), targ_poses=targs, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1, language_goal=self.lang_template)