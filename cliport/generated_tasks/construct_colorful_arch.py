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

class ConstructColorfulArch(Task):
    """Construct an arch using six blocks: three red, and three blue."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Construct an arch using six blocks: three red, and three blue."
        self.task_completed_desc = "done constructing colorful arch."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        colors = [utils.COLORS['red'], utils.COLORS['blue']]
        blocks = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            color = colors[i // 3]  # First three blocks are red, last three are blue
            block_id = env.add_object(block_urdf, block_pose, color=color)
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.02), (0, 0.05, 0.02),  # Base layer
                     (0, 0, 0.06),  # Second layer
                     (0, -0.05, 0.10), (0, 0.05, 0.10),  # Third layer
                     (0, 0, 0.14)]  # Top layer
        targs = [(utils.apply(block_pose, i), block_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in an arch (bottom layer: red, red).
        self.add_goal(objs=blocks[:2], matches=np.ones((2, 2)), targ_poses=targs[:2], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*2,
                          language_goal="Place two red blocks on the tabletop parallel to each other")

        # Goal: blocks are stacked in an arch (second layer: blue).
        self.add_goal(objs=blocks[2:3], matches=np.ones((1, 1)), targ_poses=targs[2:3], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2],
                          language_goal="Place a blue block on top of the red blocks to form a basic arch")

        # Goal: blocks are stacked in an arch (third layer: red, red).
        self.add_goal(objs=blocks[3:5], matches=np.ones((2, 2)), targ_poses=targs[3:5], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*2,
                          language_goal="Place a red block on each side of the base arch")

        # Goal: blocks are stacked in an arch (top layer: blue).
        self.add_goal(objs=blocks[5:], matches=np.ones((1, 1)), targ_poses=targs[5:], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2],
                          language_goal="Bridge them with the last blue block")