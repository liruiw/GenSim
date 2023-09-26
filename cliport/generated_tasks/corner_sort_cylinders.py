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

class CornerSortCylinders(Task):
    """Pick up cylinders of four different colors (red, blue, green, yellow) and place them into four corners accordingly marked on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} cylinder in the {color} corner"
        self.task_completed_desc = "done sorting cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors
        colors = ['red', 'blue', 'green', 'yellow']

        # Add corners
        corner_size = (0.04, 0.04, 0.04)  # x, y, z dimensions for the asset size
        corner_template = 'corner/corner-template.urdf'
        corner_poses = []
        for color in colors:
            replace = {'DIM': corner_size, 'HALF': (corner_size[0] / 2, corner_size[1] / 2, corner_size[2] / 2), 'COLOR': utils.COLORS[color]}
            corner_urdf = self.fill_template(corner_template, replace)
            corner_pose = self.get_random_pose(env, corner_size)
            env.add_object(corner_urdf, corner_pose, 'fixed')
            corner_poses.append(corner_pose)

        # Add cylinders
        cylinder_size = (0.02, 0.02, 0.06)  # x, y, z dimensions for the asset size
        cylinder_template = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for color in colors:
            replace = {'DIM': cylinder_size, 'HALF': (cylinder_size[0] / 2, cylinder_size[1] / 2, cylinder_size[2] / 2), 'COLOR': utils.COLORS[color]}
            cylinder_urdf = self.fill_template(cylinder_template, replace)
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose)
            cylinders.append(cylinder_id)

        # Add goals
        for i in range(len(cylinders)):
            self.add_goal(objs=[cylinders[i]], matches=np.int32([[1]]), targ_poses=[corner_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(cylinders),
                          language_goal=self.lang_template.format(color=colors[i]))