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

class SphereAlignStand(Task):
    """Pick up each sphere and place it on the stand of the matching color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 5
        self.lang_template = "place the {color} sphere on the {color} stand"
        self.task_completed_desc = "done aligning spheres with stands."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for the spheres and stands
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        color_names = ['red', 'green', 'blue', 'yellow', 'purple']

        # Add stands.
        # x, y, z dimensions for the asset size
        stand_size = (0.05, 0.05, 0.05)
        stand_urdf = 'stacking/stand.urdf'
        stand_poses = []
        for i in range(5):
            stand_pose = self.get_random_pose(env, stand_size)
            env.add_object(stand_urdf, stand_pose, 'fixed', color=utils.COLORS[colors[i]])
            stand_poses.append(stand_pose)

        # Add spheres.
        # x, y, z dimensions for the asset size
        sphere_size = (0.04, 0.04, 0.04)
        sphere_urdf = 'sphere/sphere.urdf'
        spheres = []
        for i in range(5):
            sphere_pose = self.get_random_pose(env, sphere_size)
            sphere_id = env.add_object(sphere_urdf, sphere_pose, color=utils.COLORS[colors[i]])
            spheres.append(sphere_id)

        # Goal: each sphere is on the stand of the matching color.
        for i in range(5):
            self.add_goal(objs=[spheres[i]], matches=np.ones((1, 1)), targ_poses=[stand_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/5,
                          language_goal=self.lang_template.format(color=color_names[i]))