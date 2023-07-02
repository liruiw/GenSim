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

class ColorLinkedBallBowlOrdering(Task):
    """Pick up each ball and place it in the bowl of the same color, in the specific sequence of red, blue, green and yellow from left to right."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} ball in the {color} bowl"
        self.task_completed_desc = "done placing balls in bowls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their order
        colors = ['red', 'blue', 'green', 'yellow']

        # Add bowls.
        # x, y, z dimensions for the asset size
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i in range(4):
            bowl_pose = self.get_random_pose(env, bowl_size)
            env.add_object(bowl_urdf, bowl_pose, 'fixed', color=utils.COLORS[colors[i]])
            bowl_poses.append(bowl_pose)

        # Add balls.
        # x, y, z dimensions for the asset size
        balls = []
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball.urdf'
        for i in range(4):
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=utils.COLORS[colors[i]])
            balls.append(ball_id)

        # Goal: each ball is in the bowl of the same color.
        for i in range(4):
            self.add_goal(objs=[balls[i]], matches=np.int32([[1]]), targ_poses=[bowl_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4)
            self.lang_goals.append(self.lang_template.format(color=colors[i]))