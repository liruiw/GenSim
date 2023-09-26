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

class CylinderBalancingAndPlacement(Task):
    """Pick up each cylinder and balance it on its end at the center of the corresponding colored zone."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "balance the {color} cylinder in the {color} zone"
        self.task_completed_desc = "done balancing and placing cylinders."
        self.colors = ['red', 'green', 'blue']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        for color in self.colors:
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[color])
            zone_poses.append(zone_pose)

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.12)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for color in self.colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[color])
            cylinders.append(cylinder_id)

        # Goal: each cylinder is balanced in the corresponding colored zone.
        for i in range(len(cylinders)):
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[zone_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/3,
                          language_goal=self.lang_template.format(color=self.colors[i]))