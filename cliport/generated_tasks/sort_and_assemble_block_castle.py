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

class SortAndAssembleBlockCastle(Task):
    """Sort blocks by color and assemble them into a castle-like structure."""

    def __init__(self):
        super().__init__()
        self.max_steps = 50
        self.lang_template = "sort the blocks by color and assemble them into a castle"
        self.task_completed_desc = "done sorting and assembling."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        for _ in range(3):
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed')
            zone_poses.append(zone_pose)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['red'], utils.COLORS['green'], utils.COLORS['blue']]
        blocks = []
        for color in block_colors:
            for _ in range(4):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=color)
                blocks.append(block_id)

        # Goal: each block is in a different zone based on color.
        for i in range(12):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 3)), targ_poses=zone_poses, replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/12)

        # Goal: blocks are stacked in a pyramid in each zone.
        for i in range(3):
            zone_blocks = blocks[i*4:(i+1)*4]
            place_pos = [(0, -0.02, 0.02), (0, 0.02, 0.02),
                         (0, 0, 0.06), (0, 0, 0.10)]
            targs = [(utils.apply(zone_poses[i], pos), zone_poses[i][1]) for pos in place_pos]
            for j in range(4):
                self.add_goal(objs=[zone_blocks[j]], matches=np.ones((1, 1)), targ_poses=[targs[j]], replace=False,
                              rotations=True, metric='pose', params=None, step_max_reward=1/12, language_goal=self.lang_template)