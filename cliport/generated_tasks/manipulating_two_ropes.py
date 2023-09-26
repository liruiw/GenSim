import numpy as np
import os
import pybullet as p
import random

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula

class ManipulatingTwoRopes(Task):
    """rearrange the red and blue deformable ropes such that it connects the two endpoints of a 3-sided square of corresponding color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "rearrange the {color_name} rope such that it connects the two endpoints of a 3-sided square of corresponding color."
        self.task_completed_desc = "done manipulating two ropes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        n_parts = 20
        radius = 0.005
        length = 2 * radius * n_parts * np.sqrt(2)

        # Add 3-sided square for the red rope.
        color_list = ['red', 'blue']
        for color_name in color_list:
            square_size = (length, length, 0)
            square_pose = self.get_random_pose(env, square_size)
            square_template = 'square/square-template.urdf'

            # IMPORTANT: REPLACE THE TEMPLATE URDF  with `fill_template`
            replace = {'DIM': (length,), 'HALF': (np.float32(length) / 2 - 0.005,)}
            urdf = self.fill_template(square_template, replace)
            env.add_object(urdf, square_pose, 'fixed', color=utils.COLORS[color_name])

            # compute corners
            corner0 = (length / 2, length / 2, 0.001)
            corner1 = (-length / 2, length / 2, 0.001)
            corner_0 = utils.apply(square_pose, corner0)
            corner_1 = utils.apply(square_pose, corner1)

            # IMPORTANT: use `make_ropes` to add cable (series of articulated small blocks).
            objects, targets, matches = self.make_ropes(env, corners=(corner_0, corner_1), color_name=color_name)
            self.add_goal(objs=objects, matches=matches, targ_poses=targets, replace=False,
                    rotations=False, metric='pose', params=None, step_max_reward=1. / len(color_list),
                          language_goal=self.lang_template.format(color_name=color_name))

        print(f"len of languages: {len(self.lang_goals)} obj:{len(objects)}")
        for i in range(480):
            p.stepSimulation()
