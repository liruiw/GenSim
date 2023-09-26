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

class ColorSequencedBlockInContainer(Task):
    """Pick up each block and place it in the container in a specific color sequence - red first, then blue, followed by green, then yellow and finally orange."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "put the {color} block in the container"
        self.task_completed_desc = "done placing blocks in container."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add container.
        container_size = (0.15, 0.15, 0.05)
        container_pose = self.get_random_pose(env, container_size)
        container_urdf = 'container/container-template.urdf'
        env.add_object(container_urdf, container_pose, 'fixed')

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['orange']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(5):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # Goal: each block is in the container in the color sequence.
        for i in range(5):
            language_goal = self.lang_template.format(color=list(utils.COLORS.keys())[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[container_pose], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 5, language_goal=language_goal)