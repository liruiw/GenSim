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

class ColorCodedBlocksOnCorner(Task):
    """Pick up blocks of different colors and place them in a corner structure in a specific color sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the blocks in the corner in the sequence red, blue, green, yellow"
        self.task_completed_desc = "done placing blocks in the corner."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add corner structure.
        corner_size = (0.15, 0.15, 0.05)
        corner_pose = self.get_random_pose(env, corner_size)
        corner_urdf = 'corner/corner-template.urdf'
        env.add_object(corner_urdf, corner_pose, 'fixed')

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, 0, 0.08)]
        targs = [(utils.apply(corner_pose, i), corner_pose[1]) for i in place_pos]

        # Goal: blocks are placed in the corner in the sequence red, blue, green, yellow.
        for i in range(4):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 4, 
                    language_goal=self.lang_template.format(blocks="the red, blue, green, yellow blocks"))