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

class ColorCoordinatedSphereCylinderMatching(Task):
    """Pick up each colored sphere and place it into the cylinder of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 4
        self.lang_template = "put the {} sphere into the {} cylinder"
        self.task_completed_desc = "done matching spheres and cylinders."
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add spheres and cylinders.
        sphere_size = (0.05, 0.05, 0.05)
        cylinder_size = (0.06, 0.06, 0.1)
        sphere_urdf = 'sphere/sphere.urdf'
        cylinder_urdf = 'cylinder/cylinder-template.urdf'

        objects = []
        for color in self.colors:
            # Add sphere
            sphere_pose = self.get_random_pose(env, sphere_size)
            sphere_id = env.add_object(sphere_urdf, sphere_pose, color=utils.COLORS[color])
            objects.append(sphere_id)

            # Add cylinder
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[color])
            objects.append(cylinder_id)

            # Add goal
            language_goal = self.lang_template.format(color, color)
            self.add_goal(objs=[sphere_id], matches=np.ones((1, 1)), targ_poses=[cylinder_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 4, language_goal=language_goal)