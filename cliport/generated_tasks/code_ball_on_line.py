import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os
import copy

class CodeBallOnLine(Task):
    """Arrange a set of colored blocks (red, blue, green, yellow, and orange) in a line, with the red block at one end, the blue block in the middle, the green block on top of the blue block, the yellow block on top of the green block, and the orange block at the other end."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {} ball on the {} line"
        self.task_completed_desc = "done aligning rainbow."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their order
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']

        # Add lines for each color
        line_size = (0.3, 0.0, 0.0)
        line_urdf = 'line/line-template.urdf'
        line_poses = []
        for color in colors:
            line_pose = self.get_random_pose(env, line_size)
            env.add_object(line_urdf, line_pose, 'fixed', color=color)
            line_poses.append(line_pose)

        # Add balls for each color
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        balls = []
        for color in colors:
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=color)
            balls.append(ball_id)

        # Add goals
        for i in range(len(balls)):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[line_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/len(balls),
                          language_goal=self.lang_template.format(colors[i], colors[i]))