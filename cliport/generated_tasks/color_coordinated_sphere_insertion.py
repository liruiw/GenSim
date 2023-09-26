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

class ColorCoordinatedSphereInsertion(Task):
    """Insert each sphere into the bowl of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert each sphere into the bowl of the same color"
        self.task_completed_desc = "done inserting spheres into bowls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
        colors = ['red', 'blue', 'green', 'yellow']
        color_values = [utils.COLORS[color] for color in colors]

        # Add bowls.
        # x, y, z dimensions for the asset size
        bowl_size = (0.12, 0.12, 0.02)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i in range(4):
            bowl_pose = self.get_random_pose(env, bowl_size)
            env.add_object(bowl_urdf, bowl_pose, 'fixed', color=color_values[i])
            bowl_poses.append(bowl_pose)

        # Add spheres.
        # x, y, z dimensions for the asset size
        sphere_size = (0.04, 0.04, 0.04)
        sphere_template = 'sphere/sphere-template.urdf'
        spheres = []
        for i in range(4):
            sphere_pose = self.get_random_pose(env, sphere_size)
            replace = {'DIM': sphere_size, 'HALF': (sphere_size[0] / 2, sphere_size[1] / 2, sphere_size[2] / 2)}
            sphere_urdf = self.fill_template(sphere_template, replace)
            sphere_id = env.add_object(sphere_urdf, sphere_pose, color=color_values[i])
            spheres.append(sphere_id)

        # Goal: each sphere is in a bowl of the same color.
        for i in range(4):
            self.add_goal(objs=[spheres[i]], matches=np.ones((1, 1)), targ_poses=[bowl_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4,
                          language_goal=f"insert the {colors[i]} sphere into the {colors[i]} bowl")