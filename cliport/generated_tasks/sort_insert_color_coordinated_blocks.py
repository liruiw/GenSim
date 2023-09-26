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

class SortInsertColorCoordinatedBlocks(Task):
    """Sort blocks by their colors and place them into the containers of the matching color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "sort the blocks by their colors and place them into the containers of the matching color"
        self.task_completed_desc = "done sorting and inserting blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add containers.
        container_size = (0.12, 0.12, 0.12)
        container_size = (0.1, 0.1, 0.1)
        container_pose = self.get_random_pose(env, container_size)
        container_urdf = 'container/container-template.urdf'
        replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
        container_urdf = self.fill_template(container_urdf, replace)
        container_colors = ['red', 'blue', 'green']
        container_poses = []
        for color in container_colors:
            container_pose = self.get_random_pose(env, container_size)
            env.add_object(container_urdf, container_pose, color=utils.COLORS[color])
            container_poses.append(container_pose)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = ['red', 'red', 'blue', 'blue', 'green', 'green']
        blocks = []
        for color in block_colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Goal: each block is in a container of the same color.
        for i in range(len(blocks)):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[container_poses[i//2]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/len(blocks), language_goal=self.lang_template)