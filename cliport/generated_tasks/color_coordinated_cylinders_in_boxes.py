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

class ColorCoordinatedCylindersInBoxes(Task):
    """Arrange four cylinders of different colors (red, blue, green, and yellow) into four matching colored boxes, such that each box contains a cylinder of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {color} cylinder in the {color} box"
        self.task_completed_desc = "done arranging cylinders in boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']]
        color_names = ['red', 'blue', 'green', 'yellow']

        # Add boxes.
        box_size = (0.05, 0.05, 0.05)  # x, y, z dimensions for the box size
        box_urdf = 'box/box-template.urdf'
        boxes = []
        for i in range(4):
            box_pose = self.get_random_pose(env, box_size)
            box_id = env.add_object(box_urdf, box_pose, color=colors[i])
            boxes.append(box_id)

        # Add cylinders.
        cylinder_size = (0.02, 0.02, 0.05)  # x, y, z dimensions for the cylinder size
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=colors[i])
            cylinders.append(cylinder_id)

        # Goal: each cylinder is in a box of the same color.
        for i in range(4):
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[box_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 4)
            self.lang_goals.append(self.lang_template.format(color=color_names[i]))