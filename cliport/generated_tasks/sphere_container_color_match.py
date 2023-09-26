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
import pybullet as p

class SphereContainerColorMatch(Task):
    """Pick up each sphere and place it into a container of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 4
        self.lang_template = "put the {color} sphere in the {color} container"
        self.task_completed_desc = "done matching spheres and containers."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']]
        color_names = ['red', 'blue', 'green', 'yellow']

        # Add containers.
        container_size = (0.12, 0.12, 0.12)
        container_urdf = 'container/container-template.urdf'
        containers = []
        for i in range(4):
            container_pose = self.get_random_pose(env, container_size)
            container_id = env.add_object(container_urdf, container_pose, color=colors[i])
            containers.append(container_id)

        # Add spheres.
        sphere_size = (0.04, 0.04, 0.04)
        sphere_urdf = 'sphere/sphere.urdf'
        spheres = []
        for i in range(4):
            sphere_pose = self.get_random_pose(env, sphere_size)
            sphere_id = env.add_object(sphere_urdf, sphere_pose, color=colors[i])
            spheres.append(sphere_id)

        # Goal: each sphere is in a container of the same color.
        for i in range(4):
            self.add_goal(objs=[spheres[i]], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(containers[i])], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/4,
                    language_goal=self.lang_template.format(color=color_names[i]))