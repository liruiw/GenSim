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
import pybullet as p

class MultiLevelInsertionAndZoneMatching(Task):
    """Pick up ell objects from their current position and insert them into the corresponding colored zone on the same level, in a specific order - large, medium, and small."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert the {size} {color} ell into the {color} zone on the same level"
        self.task_completed_desc = "done inserting."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_sizes = [(0.12, 0.12, 0), (0.12, 0.12, 0.05), (0.12, 0.12, 0.1)]
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        zone_colors = ['red', 'blue', 'green']
        for i in range(3):
            zone_pose = self.get_random_pose(env, zone_sizes[i])
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[zone_colors[i]])
            zone_poses.append(zone_pose)

        # Add ell objects.
        ell_sizes = [(0.08, 0.08, 0.02), (0.06, 0.06, 0.015), (0.04, 0.04, 0.01)]
        ell_urdf = 'insertion/ell.urdf'
        ells = []
        for i in range(3):
            for j in range(3):
                ell_pose = self.get_random_pose(env, ell_sizes[j])
                ell_id = env.add_object(ell_urdf, ell_pose, color=utils.COLORS[zone_colors[i]])
                ells.append(ell_id)

        # Goal: each ell object is in the corresponding colored zone on the same level.
        for i in range(9):
            self.add_goal(objs=[ells[i]], matches=np.ones((1, 1)), targ_poses=[zone_poses[i//3]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/9,
                          language_goal=self.lang_template.format(size=['large', 'medium', 'small'][i%3], color=zone_colors[i//3]))