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

class ColorSpecificContainerFill(Task):
    """Arrange four colored blocks (red, blue, green, and yellow) around a pallet. 
    Then, pick up these blocks and place them inside a container marked in the same color. 
    The task requires precise placement, color matching, and an understanding of spatial structures."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {color} block in the {color} container"
        self.task_completed_desc = "done arranging blocks in containers."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        # x, y, z dimensions for the asset size
        pallet_size = (0.15, 0.15, 0.01)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Define block and container colors.
        colors = ['red', 'blue', 'green', 'yellow']

        # Add blocks and containers.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        container_size = (0.12, 0.12, 0.05)
        container_template = 'container/container-template.urdf'
        blocks = []
        containers = []
        for color in colors:
            # Add block.
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

            # Add container.
            container_pose = self.get_random_pose(env, container_size)
            replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
            container_urdf = self.fill_template(container_template, replace)
            container_id = env.add_object(container_urdf, container_pose, 'fixed', color=utils.COLORS[color])
            containers.append(container_id)

        # Goal: each block is in a container of the same color.
        for i in range(len(colors)):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(containers[i])], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / len(colors),
                          language_goal=self.lang_template.format(color=colors[i]))