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

class CylinderCornerInsert(Task):
    """Place each colored cylinder into the corresponding color-marked corner of the table."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} cylinder in the {color} corner"
        self.task_completed_desc = "done placing cylinders in corners."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Cylinder colors.
        colors = ['red', 'blue', 'green', 'yellow']

        # Add corners.
        corner_size = (0.05, 0.05, 0.005)
        corner_urdf = 'corner/corner-template.urdf'
        corner_poses = []
        for i in range(4):
            corner_pose = self.get_random_pose(env, corner_size)
            env.add_object(corner_urdf, corner_pose, category='fixed', color=utils.COLORS[colors[i]])
            corner_poses.append(corner_pose)

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[colors[i]])
            cylinders.append(cylinder_id)

        # Goal: each colored cylinder is in the corresponding colored corner.
        for i in range(4):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[corner_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4, language_goal=language_goal)