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

class SweepAndSortBlocks(Task):
    """Sweep a pile of small blocks of different colors (red, blue, green, and yellow) into their corresponding colored zones marked on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "sweep the pile of {color} blocks into the {color} square"
        self.task_completed_desc = "done sweeping and sorting."
        self.primitive = primitives.push
        self.ee = Spatula
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add colored zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        colors = ['red', 'blue', 'green', 'yellow']
        zone_poses = []
        for color in colors:
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[color])
            zone_poses.append(zone_pose)

        # Add piles of colored blocks.
        block_urdf = 'block/small.urdf'
        block_size = (0.04, 0.04, 0.04)
        piles = []
        for color in colors:
            pile = []
            for _ in range(10):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
                pile.append(block_id)
            piles.append(pile)

        # Add goals for each color.
        for i, color in enumerate(colors):
            self.add_goal(objs=piles[i], matches=np.ones((10, 1)), targ_poses=[zone_poses[i]], replace=True,
                          rotations=False, metric='zone', params=[(zone_poses[i], zone_size)], step_max_reward=1,
                          language_goal=self.lang_template.format(color=color))