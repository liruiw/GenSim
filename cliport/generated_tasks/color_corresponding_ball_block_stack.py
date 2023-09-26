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

class ColorCorrespondingBallBlockStack(Task):
    """Arrange five colored balls (red, blue, green, yellow, orange) and five matching colored blocks on the tabletop, then stack the blocks in a tower on the pallet, followed by placing the corresponding colored ball on top of each block, the sequence from bottom to top should be red, blue, green, yellow, and orange."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack the blocks in a tower on the pallet in the sequence of red, blue, green, yellow, and orange, then place the corresponding colored ball on top of each block"
        self.task_completed_desc = "done stacking blocks and balls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their order
        colors = ['red', 'blue', 'green', 'yellow', 'orange']

        # Add blocks and balls
        block_size = (0.04, 0.04, 0.04)
        ball_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        ball_urdf = 'ball/ball-template.urdf'
        blocks = []
        balls = []
        for color in colors:
            block_pose = self.get_random_pose(env, block_size)
            ball_pose = self.get_random_pose(env, ball_size)
            block_id = env.add_object(block_urdf, block_pose, color=color)
            ball_id = env.add_object(ball_urdf, ball_pose, color=color)
            blocks.append(block_id)
            balls.append(ball_id)

        # Add pallet
        pallet_size = (0.15, 0.15, 0.05)
        pallet_pose = self.get_random_pose(env, pallet_size)
        pallet_urdf = 'pallet/pallet.urdf'
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Goal: each block is stacked on the pallet in the color order, and each ball is placed on the corresponding block
        for i in range(len(colors)):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[pallet_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(colors),
                          language_goal=self.lang_template)
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[pallet_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(colors),
                          language_goal=self.lang_template)