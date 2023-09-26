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

class ColorCoordinatedBlockPyramidConstruction(Task):
    """Construct two pyramids using six blocks of two different colors (three red and three blue) in two separate zones marked on the tabletop. The bottom layer should contain two blocks of the same color, followed by the second layer of one block. The pyramid in the left zone should be red and the one in the right zone should be blue. The task requires careful placement and color matching."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "build a pyramid of {color} blocks in the {zone} zone"
        self.task_completed_desc = "done building pyramids."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_size = (0.15, 0.15, 0)
        zone_urdf = 'zone/zone.urdf'
        zone1_pose = self.get_random_pose(env, zone_size)
        zone2_pose = self.get_random_pose(env, zone_size)
        env.add_object(zone_urdf, zone1_pose, 'fixed')
        env.add_object(zone_urdf, zone2_pose, 'fixed')

        # Block colors.
        colors = [utils.COLORS['red'], utils.COLORS['blue']]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i//3])
            blocks.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0.05, 0.03), (0, 0, 0.08)]
        targs1 = [(utils.apply(zone1_pose, i), zone1_pose[1]) for i in place_pos]
        targs2 = [(utils.apply(zone2_pose, i), zone2_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid in the left zone (red).
        language_goal = self.lang_template.format(color="red", zone="left")
        self.add_goal(objs=blocks[:3], matches=np.ones((3, 3)), targ_poses=targs1, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3, language_goal=language_goal)

        # Goal: blocks are stacked in a pyramid in the right zone (blue).
        language_goal = self.lang_template.format(color="blue", zone="right")
        self.add_goal(objs=blocks[3:], matches=np.ones((3, 3)), targ_poses=targs2, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3, language_goal=language_goal)