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

class ColorCoordinatedContainerFill(Task):
    """Arrange four colored blocks (red, blue, green, and yellow) around a pallet. 
    Then, pick up these blocks and place them inside a container marked in the same color. 
    The task requires precise placement, color matching, and an understanding of spatial structures."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {color} block in the {color} container"
        self.task_completed_desc = "done placing blocks in containers."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']]
        color_names = ['red', 'blue', 'green', 'yellow']

        # Add pallet.
        pallet_size = (0.3, 0.3, 0.05)
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object('pallet/pallet.urdf', pallet_pose, 'fixed')

        # Add blocks and containers.
        block_size = (0.04, 0.04, 0.04)
        container_size = (0.12, 0.12, 0.12)
        container_template = 'container/container-template.urdf'
        blocks = []
        containers = []
        for i in range(4):
            # Add block
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object('block/block.urdf', block_pose, color=colors[i])
            blocks.append(block_id)

            # Add container
            container_pose = self.get_random_pose(env, container_size)
            replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
            container_urdf = self.fill_template(container_template, replace)
            container_id = env.add_object(container_urdf, container_pose, 'fixed', color=colors[i])
            containers.append(container_id)

            language_goal = self.lang_template.format(color=color_names[i])
            self.add_goal(objs=[blocks[i]], matches=np.int32([[1]]), targ_poses=[container_pose], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/4, language_goal=language_goal)