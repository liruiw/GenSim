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

class ColorCoordinatedSphereAndCylinderAssembly(Task):
    """Pick up each sphere and place it on top of the cylinder of the same color, in a specific color sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} sphere on the {color} cylinder"
        self.task_completed_desc = "done placing spheres on cylinders."
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.color_sequence = ['red', 'blue', 'green', 'yellow']

    def reset(self, env):
        super().reset(env)

        # Add spheres and cylinders.
        sphere_size = (0.05, 0.05, 0.05)
        cylinder_size = (0.05, 0.05, 0.1)
        sphere_template = 'sphere/sphere-template.urdf'
        cylinder_template = 'cylinder/cylinder-template.urdf'

        # Add spheres and cylinders of each color.
        for color in self.colors:
            sphere_pose = self.get_random_pose(env, sphere_size)
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            sphere_id = env.add_object(sphere_template, sphere_pose, color=color)
            cylinder_id = env.add_object(cylinder_template, cylinder_pose, color=color)

            # Goal: each sphere is on top of the cylinder of the same color.
            self.add_goal(objs=[sphere_id], matches=np.ones((1, 1)), targ_poses=[cylinder_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4,
                          language_goal=self.lang_template.format(color=color))

        # The task is completed in a specific color sequence.
        self.color_sequence = ['red', 'blue', 'green', 'yellow']