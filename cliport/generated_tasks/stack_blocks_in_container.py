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

class StackBlocksInContainer(Task):
    """Pick up five blocks of different colors (red, blue, green, yellow, and orange) 
    and stack them in a container in a specific sequence. 
    The bottom of the stack should start with a red block followed by a blue, 
    green, yellow and finally an orange block at the top."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "stack the blocks in the container in the following order: {order}"
        self.task_completed_desc = "done stacking blocks in container."
        self.order = ['red', 'blue', 'green', 'yellow', 'orange']
        self.colors = [utils.COLORS[color] for color in self.order]

    def reset(self, env):
        super().reset(env)

        # Add container.
        container_size = (0.15, 0.15, 0.15)  # x, y, z dimensions for the container size
        container_pose = self.get_random_pose(env, container_size)
        container_urdf = 'container/container-template.urdf'
        replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
        container_urdf = self.fill_template(container_urdf, replace)
        env.add_object(container_urdf, container_pose, 'fixed')

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)  # x, y, z dimensions for the block size
        block_urdf = 'block/block.urdf'
        blocks = []
        for color in self.colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=color)
            blocks.append(block_id)

        # Goal: each block is stacked in the container in the specified order.
        for i in range(len(blocks)):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[container_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(blocks),
                            language_goal=self.lang_template.format(order=', '.join(self.order)))