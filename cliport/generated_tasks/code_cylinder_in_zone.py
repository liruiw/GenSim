import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os
import copy

class CodeCylinderInZone(Task):
    """Arrange a cylinder in a zone marked by a green box on the tabletop."""
    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Arrange the cylinder in the green zone"
        self.task_completed_desc = "done arranging."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zone box.
        zone_size = (0.12, 0.12, 0.02)
        zone_pose = self.get_random_pose(env, zone_size)
        zone_urdf = 'zone/zone.urdf'
        env.add_object(zone_urdf, zone_pose, 'fixed')

        # Add cylinder.
        cylinder_size = (0.04, 0.04, 0.12)
        cylinder_pose = self.get_random_pose(env, cylinder_size)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinder_id = env.add_object(cylinder_urdf, cylinder_pose)

        # Add goal.
        self.add_goal(objs=[cylinder_id], matches=np.ones((1, 1)), targ_poses=[zone_pose], replace=True,
                rotations=True, metric='pose', params=None, step_max_reward=1,
                language_goal=self.lang_template)