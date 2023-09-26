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

class ColoredCylinderZoneSorting(Task):
    """Pick up cylinders of different colors and place them into the zone of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "place the {color} cylinder in the {color} zone"
        self.task_completed_desc = "done sorting cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define cylinder and zone colors
        colors = ['red', 'blue', 'green']
        color_names = ['red', 'blue', 'green']

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        for color in colors:
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=color)
            zone_poses.append(zone_pose)

        # Add cylinders.
        cylinders = []
        cylinder_size = (0.04, 0.04, 0.12)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        for color in colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=color)
            cylinders.append(cylinder_id)

        # Add small blocks as obstacles.
        block_size = (0.02, 0.02, 0.02)
        block_urdf = 'block/small.urdf'
        for _ in range(5):
            block_pose = self.get_random_pose(env, block_size)
            env.add_object(block_urdf, block_pose)

        # Goal: each cylinder is in the zone of the same color.
        for i in range(len(cylinders)):
            language_goal = self.lang_template.format(color=color_names[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[zone_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(cylinders),
                          language_goal=language_goal)