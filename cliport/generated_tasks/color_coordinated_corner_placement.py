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

class ColorCoordinatedCornerPlacement(Task):
    """Pick up each cylinder and place it into the corner of the same color in a specific sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {} cylinder in the {} corner"
        self.task_completed_desc = "done placing cylinders in corners."
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add corners.
        corner_size = (0.15, 0.15, 0.01)
        corner_urdf = 'corner/corner-template.urdf'
        corner_poses = []
        for i in range(4):
            corner_pose = self.get_random_pose(env, corner_size)
            color = utils.COLORS[self.colors[i]]
            env.add_object(corner_urdf, corner_pose, 'fixed', color=color)
            corner_poses.append(corner_pose)

        # Add cylinders.
        cylinder_size = (0.03, 0.03, 0.08)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            color = utils.COLORS[self.colors[i]]
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=color)
            cylinders.append(cylinder_id)

        # Goal: each cylinder is in the corner of the same color.
        for i in range(4):
            language_goal = self.lang_template.format(self.colors[i], self.colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[corner_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4, language_goal=language_goal)