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

class ConstructCornerBlocks(Task):
    """Create a corner structure using four blocks. Two red blocks form the base, one on each side of the corner, followed by a green block that is positioned on the red blocks at the corner junction, and finally a blue block on top of the green one. The overall structure forms a 3-D corner."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Create a corner structure using four blocks. Two red blocks form the base, one on each side of the corner, followed by a green block that is positioned on the red blocks at the corner junction, and finally a blue block on top of the green one. The overall structure forms a 3-D corner."
        self.task_completed_desc = "done constructing corner blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add corner.
        corner_size = (0.15, 0.15, 0.05)
        corner_urdf = 'corner/corner-template.urdf'
        corner_pose = self.get_random_pose(env, corner_size)
        env.add_object(corner_urdf, corner_pose, 'fixed')

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['red'], utils.COLORS['red'], utils.COLORS['green'], utils.COLORS['blue']]
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.02), (0, 0.05, 0.02), (0, 0, 0.06), (0, 0, 0.10)]
        targs = [(utils.apply(corner_pose, i), corner_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a corner (bottom row: two red blocks).
        self.add_goal(objs=blocks[:2], matches=np.ones((2, 2)), targ_poses=targs[:2], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*2,
                          language_goal=self.lang_template)

        # Goal: blocks are stacked in a corner (middle row: one green block).
        self.add_goal(objs=blocks[2:3], matches=np.ones((1, 1)), targ_poses=targs[2:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*1,
                          language_goal=self.lang_template)

        # Goal: blocks are stacked in a corner (top row: one blue block).
        self.add_goal(objs=blocks[3:], matches=np.ones((1, 1)), targ_poses=targs[3:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2]*1,
                          language_goal=self.lang_template)
