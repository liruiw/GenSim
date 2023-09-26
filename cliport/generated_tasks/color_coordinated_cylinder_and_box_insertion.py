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

class ColorCoordinatedCylinderAndBoxInsertion(Task):
    """Pick up each colored cylinder and insert it into the box of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert the {color} cylinder into the {color} box"
        self.task_completed_desc = "done inserting cylinders into boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Cylinder and box colors.
        colors = ['red', 'blue', 'yellow', 'green']

        # Add boxes.
        box_size = (0.12, 0.12, 0.12)
        box_urdf = 'box/box-template.urdf'
        box_poses = []
        for color in colors:
            box_pose = self.get_random_pose(env, box_size)
            env.add_object(box_urdf, box_pose, category='fixed', color=utils.COLORS[color])
            box_poses.append(box_pose)

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for color in colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[color])
            cylinders.append(cylinder_id)

        # Goal: each colored cylinder is in the box of the same color.
        for i in range(len(colors)):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[box_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(colors),
                          language_goal=language_goal)