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

class ColorSortedContainerStack(Task):
    """Stack four differently colored blocks (red, blue, green, yellow) inside a container."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "stack the blocks in the container in the order: red, blue, green, then yellow"
        self.task_completed_desc = "done stacking blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add container.
        # x, y, z dimensions for the asset size
        container_size = (0.15, 0.15, 0.15)
        container_pose = self.get_random_pose(env, container_size)
        container_urdf = 'container/container-template.urdf'
        replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
        container_urdf = self.fill_template(container_urdf, replace)
        env.add_object(container_urdf, container_pose, 'fixed')

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']]
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_colors[i])
            blocks.append(block_id)

        # Add bowls.
        # x, y, z dimensions for the asset size
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        for i in range(2):
            bowl_pose = self.get_random_pose(env, bowl_size)
            env.add_object(bowl_urdf, bowl_pose, 'fixed')

        # Goal: each block is stacked in the container in the order: red, blue, green, yellow.
        for i in range(4):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[container_pose], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4,
                language_goal=self.lang_template)