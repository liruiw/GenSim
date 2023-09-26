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

class ContainerBlockSorting(Task):
    """Sort four differently colored blocks (red, blue, green, yellow) into four matching colored containers. 
    The containers are initially stacked, and need to be unstacked before the blocks can be sorted into them."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "sort the {color} block into the {color} container"
        self.task_completed_desc = "done sorting blocks into containers."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for blocks and containers
        colors = ['red', 'blue', 'green', 'yellow']

        # Add containers.
        container_size = (0.12, 0.12, 0.12)
        container_urdf = 'container/container-template.urdf'
        containers = []
        for color in colors:
            container_pose = self.get_random_pose(env, container_size)
            container_id = env.add_object(container_urdf, container_pose, color=color)
            containers.append(container_id)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for color in colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=color)
            blocks.append(block_id)

        # Goal: each block is in the matching colored container.
        for i in range(len(blocks)):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[container_pose], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / len(blocks),
                language_goal=language_goal)