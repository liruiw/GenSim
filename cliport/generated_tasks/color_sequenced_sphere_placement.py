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

class ColorSequencedSpherePlacement(Task):
    """Pick up spheres of different colors and place them in the center of the square of the same color in a specific sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} sphere in the {color} square"
        self.task_completed_desc = "done placing spheres."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their sequence
        colors = ['red', 'blue', 'green', 'yellow']

        # Add squares of different colors
        square_size = (0.1, 0.1, 0.005)
        square_urdf = 'square/square-template.urdf'
        square_poses = []
        for color in colors:
            square_pose = self.get_random_pose(env, square_size)
            env.add_object(square_urdf, square_pose, 'fixed', color=color)
            square_poses.append(square_pose)

        # Add spheres of different colors
        sphere_size = (0.04, 0.04, 0.04)
        sphere_urdf = 'sphere/sphere.urdf'
        spheres = []
        for color in colors:
            sphere_pose = self.get_random_pose(env, sphere_size)
            sphere_id = env.add_object(sphere_urdf, sphere_pose, color=color)
            spheres.append(sphere_id)

        # Goal: each sphere is in the square of the same color, in the correct sequence
        for i in range(len(colors)):
            self.add_goal(objs=[spheres[i]], matches=np.ones((1, 1)), targ_poses=[square_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/len(colors),
                          language_goal=self.lang_template.format(color=colors[i]))