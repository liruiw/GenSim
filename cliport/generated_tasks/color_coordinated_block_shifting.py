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
import pybullet as p

class ColorCoordinatedBlockShifting(Task):
    """Pick up each block and precisely place it in the zone of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "move the {color} blocks to the {color} zone"
        self.task_completed_desc = "done moving blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_colors = ['yellow', 'blue', 'green']
        zone_poses = []
        for color in zone_colors:
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[color])
            zone_poses.append(zone_pose)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        blocks = []
        for color in zone_colors:
            for _ in range(3):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
                blocks.append(block_id)

        # Add small blocks as obstacles.
        small_block_size = (0.02, 0.02, 0.02)
        small_block_urdf = 'stacking/block.urdf'
        for _ in range(5):
            small_block_pose = self.get_random_pose(env, small_block_size)
            env.add_object(small_block_urdf, small_block_pose)

        # Goal: each block is in the zone of the same color.
        for i in range(9):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[zone_poses[i//3]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/9,
                          language_goal=self.lang_template.format(color=zone_colors[i//3]))