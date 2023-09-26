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

class ColorSortedBlockRace(Task):
    """Pick up blocks of two colors and place them in corresponding colored zones in a sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the blocks in the corresponding colored zones in sequence"
        self.task_completed_desc = "done placing blocks in zones."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_colors = ['blue', 'red']
        zone_poses = []
        for color in zone_colors:
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[color])
            zone_poses.append(zone_pose)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = ['blue', 'red']
        blocks = []
        for color in block_colors:
            for _ in range(3):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
                blocks.append(block_id)

        # Goal: each block is in the corresponding colored zone.
        for i, block in enumerate(blocks):
            self.add_goal(objs=[block], matches=np.ones((1, 1)), targ_poses=[zone_poses[i//3]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/len(blocks),
                          language_goal=self.lang_template)