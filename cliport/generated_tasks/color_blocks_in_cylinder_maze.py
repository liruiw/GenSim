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

class ColorBlocksInCylinderMaze(Task):
    """Pick up five differently colored blocks (red, blue, yellow, green, and orange) that are scattered randomly on the table top. Arrange three cylindrical containers in a row to create a maze-like structure. Place the red, yellow, and blue block into the first, second, and third cylinder from left respectively. Then, stack the green and orange block on top of any container, followed by placing the same color palette on the respective block."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "arrange the blocks in the cylinders and stack the green and orange blocks"
        self.task_completed_desc = "done arranging blocks in cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add cylinders.
        cylinder_size = (0.05, 0.05, 0.1)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinder_poses = []
        for _ in range(3):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            env.add_object(cylinder_urdf, cylinder_pose, 'fixed')
            cylinder_poses.append(cylinder_pose)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['yellow'], utils.COLORS['green'], utils.COLORS['orange']]
        blocks = []
        for i in range(5):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_colors[i])
            blocks.append(block_id)

        # Goal: red, yellow, and blue blocks are in the first, second, and third cylinder respectively.
        self.add_goal(objs=blocks[:3], matches=np.ones((3, 3)), targ_poses=cylinder_poses, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, language_goal=self.lang_template)

        # Goal: green and orange blocks are stacked on top of any cylinder.
        self.add_goal(objs=blocks[3:], matches=np.ones((2, 3)), targ_poses=cylinder_poses, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, language_goal=self.lang_template)