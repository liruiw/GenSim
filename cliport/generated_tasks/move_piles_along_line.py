import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils

class MovePilesAlongLine(Task):
    """Move three piles of small blocks, each pile a different color (red, blue, green), 
    along three matching colored lines to three separate zones of the same color using a spatula."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "move the piles of blocks along the lines to the matching colored zones"
        self.task_completed_desc = "done moving piles."
        self.primitive = primitives.push
        self.ee = Spatula
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add three colored lines.
        line_template = 'line/line-template.urdf'
        line_colors = ['red', 'blue', 'green']
        line_poses = []
        for color in line_colors:
            line_size = self.get_random_size(0.1, 0.15, 0.1, 0.15, 0.05, 0.05)
            line_pose = self.get_random_pose(env, line_size)
            replace = {'DIM': line_size, 'HALF': (line_size[0] / 2, line_size[1] / 2, line_size[2] / 2), 'COLOR': color}
            line_urdf = self.fill_template(line_template, replace)
            env.add_object(line_urdf, line_pose, 'fixed')
            line_poses.append(line_pose)

        # Add three colored zones.
        zone_template = 'zone/zone.urdf'
        zone_poses = []
        for color in line_colors:
            zone_size = self.get_random_size(0.1, 0.15, 0.1, 0.15, 0.05, 0.05)
            zone_pose = self.get_random_pose(env, zone_size)
            replace = {'DIM': zone_size, 'HALF': (zone_size[0] / 2, zone_size[1] / 2, zone_size[2] / 2), 'COLOR': color}
            zone_urdf = self.fill_template(zone_template, replace)
            env.add_object(zone_urdf, zone_pose, 'fixed')
            zone_poses.append(zone_pose)

        # Add three piles of small blocks.
        block_template = 'block/small.urdf'
        block_colors = ['red', 'blue', 'green']
        block_ids = []
        for color in block_colors:
            block_size = self.get_random_size(0.1, 0.15, 0.1, 0.15, 0.05, 0.05)
            block_pose = self.get_random_pose(env, block_size)
            replace = {'DIM': block_size, 'HALF': (block_size[0] / 2, block_size[1] / 2, block_size[2] / 2), 'COLOR': color}
            block_urdf = self.fill_template(block_template, replace)
            block_id = env.add_object(block_urdf, block_pose)
            block_ids.append(block_id)

        # Add goals.
        for i in range(3):
            self.add_goal(objs=[block_ids[i]], matches=np.ones((1, 1)), targ_poses=[zone_poses[i]], replace=False,
                          rotations=False, metric='zone', params=[(zone_poses[i], zone_size)], step_max_reward=1/3,
                          language_goal=self.lang_template)