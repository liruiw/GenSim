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

class AlignCylindersBox(Task):
    """Position three cylinders (red, green, blue) vertically inside a box."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "put the {color} cylinder in the box"
        self.task_completed_desc = "done aligning cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add box.
        box_size = (0.30, 0.10, 0.10)
        box_pose = self.get_random_pose(env, box_size)
        box_urdf = 'box/box-template.urdf'
        env.add_object(box_urdf, box_pose, 'fixed')

        # Cylinder colors.
        colors = ['red', 'green', 'blue']

        # Add cylinders.
        cylinder_size = (0.03, 0.03, 0.10)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(3):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[colors[i]])
            cylinders.append(cylinder_id)

        # IMPORTANT Associate placement locations for goals.
        place_pos = [(-0.10, 0, 0.05), (0, 0, 0.05), (0.10, 0, 0.05)]
        targs = [(utils.apply(box_pose, i), box_pose[1]) for i in place_pos]

        # Goal: cylinders are placed in the box in the order red, green, blue.
        for i in range(3):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 3, language_goal=language_goal)