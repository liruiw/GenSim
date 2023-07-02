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

class ColorCoordinatedContainerSorting(Task):
    """Sort blocks into containers of the same color in a specific order."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "sort the blocks into the containers of the same color in the specified order"
        self.task_completed_desc = "done sorting blocks into containers."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their orders in containers
        colors = ['red', 'blue', 'green', 'yellow']
        orders = [
            ['red', 'blue', 'green', 'yellow'],
            ['yellow', 'green', 'blue', 'red'],
            ['green', 'red', 'yellow', 'blue'],
            ['blue', 'yellow', 'red', 'green']
        ]

        # Add containers.
        container_size = (0.12, 0.12, 0.02)
        container_urdf = 'container/container-template.urdf'
        containers = []
        for i in range(4):
            container_pose = self.get_random_pose(env, container_size)
            color = utils.COLORS[colors[i]]
            container_id = env.add_object(container_urdf, container_pose, color=color)
            containers.append(container_id)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(4):
            for _ in range(4):
                block_pose = self.get_random_pose(env, block_size)
                color = utils.COLORS[colors[i]]
                block_id = env.add_object(block_urdf, block_pose, color=color)
                blocks.append(block_id)

        # Add goals.
        for i in range(4):
            for j in range(4):
                # Find the block and container of the same color
                block_id = blocks[i*4 + j]
                container_id = containers[i]
                # Define the target pose based on the order in the container
                container_pose = p.getBasePositionAndOrientation(container_id)[0]
                targ_pose = (container_pose[0], container_pose[1], container_pose[2] + block_size[2] * (j + 1))
                self.add_goal(objs=[block_id], matches=np.ones((1, 1)), targ_poses=[targ_pose], replace=False,
                              rotations=True, metric='pose', params=None, step_max_reward=1/16)

        self.lang_goals.append(self.lang_template)