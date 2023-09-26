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

class BlockPyramidWithLimitedSpace(Task):
    """Sort blocks according to color into three zones on the tabletop and construct a pyramid in each zone."""

    def __init__(self):
        super().__init__()
        self.max_steps = 50
        self.lang_template = "sort the blocks according to color into three zones and construct a pyramid in each zone"
        self.task_completed_desc = "done sorting and constructing pyramids."
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

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['green'], utils.COLORS['blue'], utils.COLORS['yellow']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for color in colors:
            for _ in range(3):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=color)
                blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(zone_pose, i), zone_pose[1]) for zone_pose in zone_poses for i in place_pos]

        # Goal: blocks are sorted and stacked in a pyramid in each zone.
        for i in range(3):
            self.add_goal(objs=blocks[i*3:(i+1)*3], matches=np.ones((3, 3)), targ_poses=targs[i*3:(i+1)*3], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*3,
                    language_goal=self.lang_template)