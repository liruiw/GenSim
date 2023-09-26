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

class ColorCoordinatedCylinderAlignment(Task):
    """Align cylinders in color-coordinated zones."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the {color} cylinder in the {color} zone"
        self.task_completed_desc = "done aligning cylinders."
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        for i in range(5):
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[self.colors[i]])
            zone_poses.append(zone_pose)

        # Add cylinders.
        cylinders = []
        cylinder_size = (0.04, 0.04, 0.12)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        for i in range(5):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[self.colors[i]])
            cylinders.append(cylinder_id)

        # Goal: each cylinder is in the zone of the same color.
        for i in range(5):
            language_goal = self.lang_template.format(color=self.colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[zone_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 5, language_goal=language_goal)