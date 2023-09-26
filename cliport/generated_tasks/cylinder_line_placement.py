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

class CylinderLinePlacement(Task):
    """Place cylinders of different colors on a line in a specific color order."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the cylinders on the line in the order of red, blue, green, yellow, orange, and purple"
        self.task_completed_desc = "done placing cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add line.
        line_size = (0.6, 0.01, 0.01)
        line_pose = self.get_random_pose(env, line_size)
        line_urdf = 'line/single-green-line-template.urdf'
        env.add_object(line_urdf, line_pose, 'fixed')

        # Cylinder colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['orange'], utils.COLORS['purple']
        ]

        # Add cylinders.
        cylinder_size = (0.02, 0.02, 0.06)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'

        objs = []
        for i in range(6):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=colors[i])
            objs.append(cylinder_id)

        # IMPORTANT Associate placement locations for goals.
        place_pos = [(0.1*i, 0, 0) for i in range(6)]
        targs = [(utils.apply(line_pose, i), line_pose[1]) for i in place_pos]

        # Goal: cylinders are placed on the line in the order of red, blue, green, yellow, orange, and purple.
        self.add_goal(objs=objs, matches=np.ones((6, 6)), targ_poses=targs, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1, language_goal=self.lang_template)