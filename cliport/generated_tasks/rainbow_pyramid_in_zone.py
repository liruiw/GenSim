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

class RainbowPyramidInZone(Task):
    """Construct a pyramid inside a marked zone with differently colored blocks (red, blue, green, yellow, and purple) arranged in the order of a rainbow, with red at the bottom and purple at the top."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "build a pyramid with {blocks} in the marked zone"
        self.task_completed_desc = "done building pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add goal zone.
        zone_size = (0.12, 0.12, 0)
        zone_pose = self.get_random_pose(env, zone_size)
        env.add_object('zone/zone.urdf', zone_pose, 'fixed')

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['orange'], utils.COLORS['yellow'],
            utils.COLORS['green'], utils.COLORS['blue'], utils.COLORS['purple']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'

        objs = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            objs.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(zone_pose, i), zone_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid (bottom row: red, orange, yellow).
        language_goal = self.lang_template.format(blocks="the red, orange and yellow blocks")
        self.add_goal(objs=objs[:3], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3, language_goal=language_goal)

        # Goal: blocks are stacked in a pyramid (middle row: green, blue).
        language_goal = self.lang_template.format(blocks="the green and blue blocks")      
        self.add_goal(objs=objs[3:5], matches=np.ones((2, 2)), targ_poses=targs[3:5], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*2, language_goal=language_goal)

        # Goal: blocks are stacked in a pyramid (top row: purple).
        language_goal = self.lang_template.format(blocks="the purple block")
        self.add_goal(objs=objs[5:], matches=np.ones((1, 1)), targ_poses=targs[5:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6, symmetries=[np.pi/2]*1, language_goal=language_goal)