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

class PlaceBallInElevatedBowl(Task):
    """Pick up a red ball and carefully place it into a bowl, which is positioned on a raised platform that is surrounded by small blocks."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the red ball in the elevated bowl"
        self.task_completed_desc = "done placing ball in bowl."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add elevated platform.
        platform_size = (0.3, 0.3, 0.05)

        # Add bowl on the platform.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_pose = self.get_random_pose(env, bowl_size)
        bowl_pose[0][2] += platform_size[2]  # place the bowl on top of the platform
        bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')

        # Add red ball.
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        ball_pose = self.get_random_pose(env, ball_size)
        ball_id = env.add_object(ball_urdf, ball_pose, color=utils.COLORS['red'])

        # Add small blocks around the platform.
        block_size = (0.02, 0.02, 0.02)
        block_urdf = 'block/small.urdf'
        for _ in range(5):
            block_pose = self.get_random_pose(env, block_size)
            env.add_object(block_urdf, block_pose)

        # Goal: the red ball is in the bowl.
        self.add_goal(objs=[ball_id], matches=np.ones((1, 1)), targ_poses=[bowl_pose], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1,
                          language_goal=self.lang_template)