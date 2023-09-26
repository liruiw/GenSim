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

class ColorCoordinatedBlockBowlSorting(Task):
    """Sort four differently colored blocks (red, blue, green, yellow) into four matching colored bowls."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {color} block in the {color} bowl"
        self.task_completed_desc = "done sorting blocks into bowls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for blocks and bowls
        colors = ['red', 'blue', 'green', 'yellow']

        # Add bowls.
        # x, y, z dimensions for the asset size
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for color in colors:
            bowl_pose = self.get_random_pose(env, bowl_size)
            env.add_object(bowl_urdf, bowl_pose, color=utils.COLORS[color], category='fixed')
            bowl_poses.append(bowl_pose)

        # Add blocks.
        # x, y, z dimensions for the asset size
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        for color in colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Goal: each block is in the bowl of the same color.
        for i in range(len(colors)):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[bowl_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(colors),
                          language_goal=language_goal)