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

class StackCylinderOnBowl(Task):
    """Stack cylinders of matching colors on top of bowls in a specific sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack the {color} cylinder on the {color} bowl"
        self.task_completed_desc = "done stacking cylinders on bowls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add bowls.
        # x, y, z dimensions for the asset size
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_colors = ['red', 'blue', 'green']
        bowl_poses = []
        for color in bowl_colors:
            bowl_pose = self.get_random_pose(env, bowl_size)
            env.add_object(urdf=bowl_urdf, pose=bowl_pose, color=utils.COLORS[color], category='fixed')
            bowl_poses.append(bowl_pose)

        # Add cylinders.
        # x, y, z dimensions for the asset size
        cylinders = []
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        for color in bowl_colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[color])
            cylinders.append(cylinder_id)

        # Goal: each cylinder is stacked on a bowl of the same color.
        for i in range(len(cylinders)):
            language_goal = self.lang_template.format(color=bowl_colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[bowl_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/3, language_goal=language_goal)