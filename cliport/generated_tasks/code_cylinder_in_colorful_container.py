import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os
import copy

class CodeCylinderInColorfulContainer(Task):
    """Pick up each cylinder and place it in the container of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} cylinder in the {color} container"
        self.task_completed_desc = "done placing cylinders in containers."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add containers.
        container_size = (0.12, 0.12, 0.12)
        container_urdf = 'container/container-template.urdf'
        container_colors = ['red', 'blue', 'green', 'yellow']
        container_poses = []

        for color in container_colors:
            container_pose = self.get_random_pose(env, container_size)
            container_id = env.add_object(container_urdf, container_pose, color=color)
            container_poses.append(container_pose)

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []

        for color in container_colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=color)
            cylinders.append(cylinder_id)

        # Add goals.
        for i in range(len(container_poses)):
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[container_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/len(container_poses),
                          language_goal=self.lang_template.format(color=container_colors[i]))