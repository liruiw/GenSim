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

class ColorCoordinatedZoneStacking(Task):
    """Pick up blocks of different colors and stack them in zones to form a pyramid."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "stack the blocks in the zones to form a pyramid"
        self.task_completed_desc = "done stacking blocks."
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
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(9):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i//3])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(zone_poses[i//3], place_pos[i%3]), zone_poses[i//3][1]) for i in range(9)]

        # Goal: blocks are stacked in a pyramid in each zone.
        for i in range(3):
            self.add_goal(objs=blocks[i*3:(i+1)*3], matches=np.ones((3, 3)), targ_poses=targs[i*3:(i+1)*3], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*3,
                          language_goal=self.lang_template.format(blocks="the red, blue and green blocks",
                                                             row="bottom"))