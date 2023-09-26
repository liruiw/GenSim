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

class ColorCoordinatedBoxBallMatching(Task):
    """Pick up each ball and place it inside the box of the same color, navigate around the barrier without knocking over any small blocks."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {color} ball in the {color} box"
        self.task_completed_desc = "done placing balls in boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for the boxes and balls
        colors = ['red', 'blue', 'green', 'yellow']

        # Add boxes.
        box_size = (0.05, 0.05, 0.05)
        box_urdf = 'box/box-template.urdf'
        box_poses = []
        for color in colors:
            box_pose = self.get_random_pose(env, box_size)
            env.add_object(box_urdf, box_pose, color=color, category='fixed')
            box_poses.append(box_pose)

        # Add balls.
        balls = []
        ball_size = (0.02, 0.02, 0.02)
        ball_urdf = 'ball/ball-template.urdf'
        for color in colors:
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=color)
            balls.append(ball_id)

        # Add small blocks as barriers.
        barrier_size = (0.01, 0.01, 0.01)
        barrier_urdf = 'block/small.urdf'
        for _ in range(10):
            barrier_pose = self.get_random_pose(env, barrier_size)
            env.add_object(barrier_urdf, barrier_pose, category='fixed')

        # Goal: each ball is in the box of the same color.
        for i in range(len(balls)):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[box_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/len(balls),
                          language_goal=self.lang_template.format(color=colors[i]))