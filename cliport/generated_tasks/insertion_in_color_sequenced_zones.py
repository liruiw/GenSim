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

class InsertionInColorSequencedZones(Task):
    """Pick up each ell and place it in the zone of the same color, in the specific sequence of red, blue, green, and yellow from left to right."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} ell in the {color} zone"
        self.task_completed_desc = "done placing ells in color sequenced zones."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Ell colors.
        colors = ['red', 'blue', 'green', 'yellow']

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        for i in range(4):
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[colors[i]])
            zone_poses.append(zone_pose)

        # Add ells.
        ell_size = (0.04, 0.04, 0.04)
        ell_urdf = 'insertion/ell.urdf'
        ells = []
        for i in range(4):
            ell_pose = self.get_random_pose(env, ell_size)
            ell_id = env.add_object(ell_urdf, ell_pose, color=utils.COLORS[colors[i]])
            ells.append(ell_id)

        # Goal: each ell is in the zone of the same color.
        for i in range(4):
            self.add_goal(objs=[ells[i]], matches=np.ones((1, 1)), targ_poses=[zone_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/4,
                          language_goal=self.lang_template.format(color=colors[i]))