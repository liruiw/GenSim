"""Aligning task."""

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import os
import pybullet as p
import random
import numpy as np
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils


class GeneratedTask(Task):
    """Move two piles of small blocks, one red and one blue, along their respective lines drawn on the tabletop into a marked zone. Each pile should be moved along a unique line and into a different zone."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "move the piles of blocks along the lines into the marked zones"
        self.task_completed_desc = "done moving piles."
        self.primitive = primitives.push
        self.ee = Spatula
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add two lines.
        line_size = (0.12, 0.01, 0.01)
        line_urdf = 'line/line-template.urdf'
        line_poses = []
        for _ in range(2):
            line_pose = self.get_random_pose(env, line_size)
            env.add_object(line_urdf, line_pose, 'fixed')
            line_poses.append(line_pose)

        # Add two zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        for _ in range(2):
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed')
            zone_poses.append(zone_pose)

        # Add two piles of blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/small.urdf'
        block_colors = [utils.COLORS['red'], utils.COLORS['blue']]
        piles = []
        for i in range(2):
            pile = self.make_piles(env, num_piles=1, num_objs=5, obj_size=block_size, obj_urdf=block_urdf, obj_color=block_colors[i])
            piles.append(pile)

        # Add goals.
        for i in range(2):
            self.add_goal(objs=piles[i], matches=np.ones((len(piles[i]), 1)), targ_poses=[zone_poses[i]], replace=False,
                rotations=False, metric='zone', params=[(zone_poses[i], zone_size)], step_max_reward=1/2)
        self.lang_goals.append(self.lang_template)
