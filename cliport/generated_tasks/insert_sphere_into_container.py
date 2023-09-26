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

class InsertSphereIntoContainer(Task):
    """Pick up a blue sphere and place it into an open container."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "pick up a blue sphere and place it into an open container"
        self.task_completed_desc = "done inserting sphere into container."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add container.
        # x, y, z dimensions for the asset size
        container_size = (0.1, 0.1, 0.1)
        container_pose = self.get_random_pose(env, container_size)
        container_template = 'container/container-template.urdf'
        replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
        # IMPORTANT: REPLACE THE TEMPLATE URDF with `fill_template`
        container_urdf = self.fill_template(container_template, replace)
        container_id = env.add_object(container_urdf, container_pose, 'fixed')

        # Add sphere.
        # x, y, z dimensions for the asset size
        sphere_size = (0.04, 0.04, 0.04)
        sphere_pose = self.get_random_pose(env, sphere_size)
        sphere_urdf = 'sphere/sphere.urdf'
        sphere_id = env.add_object(sphere_urdf, sphere_pose, color=utils.COLORS['blue'])

        # Goal: the blue sphere is in the container.
        self.add_goal(objs=[sphere_id], matches=np.ones((1, 1)), targ_poses=[container_pose], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1,
                          language_goal=self.lang_template)