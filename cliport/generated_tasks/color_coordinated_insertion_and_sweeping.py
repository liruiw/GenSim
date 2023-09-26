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

class ColorCoordinatedInsertionAndSweeping(Task):
    """Pick up each block and insert it into the ell-shaped fixture of the same color, then sweep the blocks into three separate zones marked on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert the {color} block into the {color} ell and sweep it into the {color} zone"
        self.task_completed_desc = "done inserting and sweeping."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors
        colors = ['red', 'blue', 'green']

        # Add ell-shaped fixtures and blocks
        ell_ids = []
        block_ids = []
        for color in colors:
            # Add ell
            ell_size = (0.1, 0.1, 0.1)
            ell_pose = self.get_random_pose(env, ell_size)
            ell_urdf = 'insertion/ell.urdf'
            ell_id = env.add_object(ell_urdf, ell_pose, color=utils.COLORS[color])
            ell_ids.append(ell_id)

            # Add block
            block_size = (0.04, 0.04, 0.04)
            block_pose = self.get_random_pose(env, block_size)
            block_urdf = 'block/small.urdf'
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            block_ids.append(block_id)

        # Add zones
        zone_ids = []
        for color in colors:
            zone_size = (0.12, 0.12, 0)
            zone_pose = self.get_random_pose(env, zone_size)
            zone_urdf = 'zone/zone.urdf'
            zone_id = env.add_object(zone_urdf, zone_pose, color=utils.COLORS[color])
            zone_ids.append(zone_id)

        # Add goals
        for i in range(3):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[block_ids[i]], matches=np.ones((1, 1)), targ_poses=[zone_pose], replace=False,
                          rotations=True, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1 / 3,
                          language_goal=language_goal)