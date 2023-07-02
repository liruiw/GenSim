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

class TriangleBlockArrangement(Task):
    """Arrange blocks of three different colors (red, green, and blue) in a triangular layout on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.lang_template = "Arrange blocks of three different colors (red, green, and blue) in a triangular layout on the tabletop."
        self.task_completed_desc = "done arranging blocks in a triangle."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        colors = [utils.COLORS['red'], utils.COLORS['green'], utils.COLORS['blue']]
        blocks = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i//2])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.02), (0, 0, 0.02),
                     (0, 0.05, 0.02), (0, -0.025, 0.06),
                     (0, 0.025, 0.06), (0, 0, 0.10)]
        base_pose = self.get_random_pose(env, block_size)
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a triangle (bottom row: red, middle row: green, top row: blue).
        self.add_goal(objs=blocks[:3], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3)
        self.lang_goals.append("Arrange the red blocks in a row at the base of the triangle.")

        self.add_goal(objs=blocks[3:5], matches=np.ones((2, 2)), targ_poses=targs[3:5], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*2)
        self.lang_goals.append("Arrange the green blocks in a row above the red blocks.")

        self.add_goal(objs=blocks[5:], matches=np.ones((1, 1)), targ_poses=targs[5:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2])
        self.lang_goals.append("Place the blue block at the top of the triangle.")