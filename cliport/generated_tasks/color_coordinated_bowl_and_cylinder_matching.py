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

class ColorCoordinatedBowlAndCylinderMatching(Task):
    """Pick up each colored cylinder and place it into the bowl of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {} cylinder in the {} bowl"
        self.task_completed_desc = "done matching cylinders and bowls."
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for color in self.colors:
            bowl_pose = self.get_random_pose(env, bowl_size)
            env.add_object(bowl_urdf, bowl_pose, 'fixed', color=utils.COLORS[color])
            bowl_poses.append(bowl_pose)

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.12)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for color in self.colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[color])
            cylinders.append(cylinder_id)

        # Goal: each colored cylinder is in the bowl of the same color.
        for i in range(len(self.colors)):
            language_goal = self.lang_template.format(self.colors[i], self.colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[bowl_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(self.colors),
                          language_goal=language_goal)