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

class ColorMatchingBlocksToCornerAndZone(Task):
    """Pick each block and place it on the corner of the same color first, then move it to the corresponding colored zone, following the specific color sequence: red, blue, green, and finally yellow."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {} block on the {} corner and then move it to the {} zone"
        self.task_completed_desc = "done placing blocks on corners and zones."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
        colors = {'red': utils.COLORS['red'], 'blue': utils.COLORS['blue'], 'green': utils.COLORS['green'], 'yellow': utils.COLORS['yellow']}
        color_names = list(colors.keys())

        # Add corners and zones
        corner_size = (0.05, 0.05, 0.05)
        zone_size = (0.15, 0.15, 0.05)
        corner_urdf = 'corner/corner-template.urdf'
        zone_urdf = 'zone/zone.urdf'
        corners = []
        zones = []
        for color in color_names:
            corner_pose = self.get_random_pose(env, corner_size)
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(corner_urdf, corner_pose, 'fixed', color=colors[color])
            env.add_object(zone_urdf, zone_pose, 'fixed', color=colors[color])
            corners.append(corner_pose)
            zones.append(zone_pose)

        # Add blocks
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for color in color_names:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[color])
            blocks.append(block_id)

        # Add goals
        for i in range(4):
            language_goal = self.lang_template.format(color_names[i], color_names[i], color_names[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[corners[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 8, language_goal=language_goal)

        for i in range(4):
            language_goal = self.lang_template.format(color_names[i], color_names[i], color_names[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[zones[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 8, language_goal=language_goal)