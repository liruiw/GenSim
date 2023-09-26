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

class ColorOrderedContainerArrangement(Task):
    """Arrange six containers with blocks of matching colors in a specific color order."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "arrange the containers in the color order: red, blue, green, yellow, orange, and purple"
        self.task_completed_desc = "done arranging containers."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define color order
        color_order = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']

        # Add containers and blocks
        container_template = 'container/container-template.urdf'
        container_size = (0.12, 0.12, 0.02)
        replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
        container_urdf = self.fill_template(container_template, replace)

        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        containers = []
        blocks = []
        for color in color_order:
            # Add container
            container_pose = self.get_random_pose(env, container_size)
            container_id = env.add_object(container_urdf, container_pose, color=utils.COLORS[color])
            containers.append(container_id)

            # Add block
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

            # Add subgoal to place block in container
            self.add_goal(objs=[block_id], matches=np.ones((1, 1)), targ_poses=[container_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/6,
                          language_goal=self.lang_template)

        # Add final goal to arrange containers in color order
        container_poses = [self.get_random_pose(env, container_size) for _ in color_order]
        self.add_goal(objs=containers, matches=np.eye(len(color_order)), targ_poses=container_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1, language_goal=self.lang_template)