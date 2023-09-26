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

class ColorCoordinatedBlockInContainerPlacement(Task):
    """Pick up each block and accurately place it inside the container of the same color in a specific sequence - red first, then blue, followed by green, and finally yellow."""

    def __init__(self):
        super().__init__()
        self.max_steps = 4
        self.lang_template = "place the {} block in the {} container"
        self.task_completed_desc = "done placing blocks in containers."
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add containers.
        container_size = (0.12, 0.12, 0.12)
        container_urdf = 'container/container-template.urdf'
        container_poses = []
        for color in self.colors:
            container_pose = self.get_random_pose(env, container_size)
            container_id = env.add_object(container_urdf, container_pose, color=utils.COLORS[color], category='fixed')
            container_poses.append((container_id, container_pose))

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for color in self.colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Add goals.
        for i in range(len(self.colors)):
            language_goal = self.lang_template.format(self.colors[i], self.colors[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[container_poses[i][1]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(self.colors),
                          language_goal=language_goal)