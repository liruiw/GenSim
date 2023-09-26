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

class AlignCylinderInZone(Task):
    """Place the red cylinder in the middle of the green zone, and the blue cylinder in the middle of the yellow zone, without disturbing the other colored zones."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "Place the {} cylinder in the middle of the {} zone"
        self.task_completed_desc = "done aligning cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_colors = ['green', 'yellow']
        zone_poses = []
        for color in zone_colors:
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[color])
            zone_poses.append(zone_pose)

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinder_colors = ['red', 'blue']
        cylinders = []
        for color in cylinder_colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[color])
            cylinders.append(cylinder_id)

        # Goal: each cylinder is in a different zone.
        for i in range(len(cylinders)):
            language_goal = self.lang_template.format(cylinder_colors[i], zone_colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.int32([[1]]), targ_poses=[zone_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / len(cylinders),
                language_goal=language_goal)