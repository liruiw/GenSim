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

class ColorCoordinatedContainerAndBlockMatching(Task):
    """Pick up each block and place it into the container of the same color. The sequence of placement should start with red, then blue, green, and finally yellow."""

    def __init__(self):
        super().__init__()
        self.max_steps = 4
        self.lang_template = "place the {} block in the {} container"
        self.task_completed_desc = "done placing blocks in containers."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their order
        colors = ['red', 'blue', 'green', 'yellow']

        # Add containers.
        container_size = (0.12, 0.12, 0.12)
        container_urdf = 'container/container-template.urdf'
        container_poses = []
        for color in colors:
            container_pose = self.get_random_pose(env, container_size)
            env.add_object(container_urdf, container_pose, 'fixed', color=utils.COLORS[color])
            container_poses.append(container_pose)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        for color in colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Add goals
        for i in range(len(colors)):
            language_goal = self.lang_template.format(colors[i], colors[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[container_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(colors),
                          language_goal=language_goal)