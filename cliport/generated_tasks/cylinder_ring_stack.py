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

class CylinderRingStack(Task):
    """Pick up each block and stack it on top of the corresponding colored cylinder. 
    Each cylinder and block pair should be stacked inside a differently colored container."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "stack the {color} block on the {color} cylinder in the {container_color} container"
        self.task_completed_desc = "done stacking."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for cylinders, blocks and containers
        colors = ['red', 'blue', 'green', 'yellow']
        container_colors = ['blue', 'green', 'yellow', 'red']

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[colors[i]])
            cylinders.append(cylinder_id)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[colors[i]])
            blocks.append(block_id)

        # Add containers.
        container_size = (0.12, 0.12, 0.12)
        container_urdf = 'container/container-template.urdf'
        containers = []
        for i in range(4):
            container_pose = self.get_random_pose(env, container_size)
            container_id = env.add_object(container_urdf, container_pose, color=utils.COLORS[container_colors[i]])
            containers.append(container_id)

        # Goal: each block is stacked on the corresponding colored cylinder inside a differently colored container.
        for i in range(4):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[cylinder_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4,
                          language_goal=self.lang_template.format(color=colors[i], container_color=container_colors[i]))