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

class ColoredBallsSortingInCorner(Task):
    """Pick up each ball and place it in the corner of the same color, in the specific sequence of red, blue, green and yellow, starting from the leftmost corner to the rightmost."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} ball in the {color} corner"
        self.task_completed_desc = "done sorting balls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the colors and their sequence
        colors = ['red', 'blue', 'green', 'yellow']

        # Add corners.
        corner_size = (0.12, 0.12, 0)
        corner_urdf = 'corner/corner-template.urdf'
        corner_poses = []
        for i in range(4):
            corner_pose = self.get_random_pose(env, corner_size)
            env.add_object(corner_urdf, corner_pose, 'fixed', color=utils.COLORS[colors[i]])
            corner_poses.append(corner_pose)

        # Add balls.
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        balls = []
        for i in range(4):
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=utils.COLORS[colors[i]])
            balls.append(ball_id)

        # Goal: each ball is in the corner of the same color.
        for i in range(4):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[corner_poses[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/4,
                          language_goal=self.lang_template.format(color=colors[i]))