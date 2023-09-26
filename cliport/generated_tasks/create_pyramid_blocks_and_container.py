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

class CreatePyramidBlocksAndContainer(Task):
    """Create a pyramid structure using six blocks of three different colors (two red, two green, and two blue) inside a container."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Create a pyramid structure using six blocks of three different colors (two red, two green, and two blue) inside a container."
        self.task_completed_desc = "done creating pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add container.
        # x, y, z dimensions for the asset size
        container_size = (0.3, 0.3, 0.1)
        container_pose = self.get_random_pose(env, container_size)
        container_urdf = 'container/container-template.urdf'
        replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
        container_urdf = self.fill_template(container_urdf, replace)
        env.add_object(container_urdf, container_pose, 'fixed')
        self.add_corner_anchor_for_pose(env, container_pose)


        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['red'], utils.COLORS['red'], utils.COLORS['green'], utils.COLORS['green'], utils.COLORS['blue'], utils.COLORS['blue']]
        blocks = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_colors[i])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03), (0, 0.05, 0.03), (0, -0.025, 0.08), (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(container_pose, i), container_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid (bottom row: green, green, blue).
        self.add_goal(objs=blocks[2:5], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*3,
                          language_goal=self.lang_template.format(blocks="the green and blue blocks",
                                                         row="bottom"))

        # Goal: blocks are stacked in a pyramid (middle row: red, red).
        self.add_goal(objs=blocks[:2], matches=np.ones((2, 2)), targ_poses=targs[3:5], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*2,
                          language_goal=self.lang_template.format(blocks="the red blocks",
                                                         row="middle"))

        # Goal: blocks are stacked in a pyramid (top row: blue).
        self.add_goal(objs=blocks[5:], matches=np.ones((1, 1)), targ_poses=targs[5:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2],
                          language_goal=self.lang_template.format(blocks="the blue block",
                                                         row="top"))