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

class AlignBallsInColoredBoxes(Task):
    """Align balls in colored boxes according to the color and sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {color} ball in the {color} box"
        self.task_completed_desc = "done aligning balls in boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their sequence
        colors = ['red', 'blue', 'green', 'yellow']

        # Add boxes.
        box_size = (0.12, 0.12, 0.12)
        box_urdf = 'box/box-template.urdf'
        box_poses = []
        boxes = []
        for i in range(4):
            box_pose = self.get_random_pose(env, box_size)
            box_id = env.add_object(box_urdf, box_pose, color=utils.COLORS[colors[i]])
            boxes.append(box_id)
            box_poses.append(box_pose)

        # Add balls.
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball.urdf'
        balls = []
        for i in range(4):
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=utils.COLORS[colors[i]])
            balls.append(ball_id)

        # Goal: each ball is in the box of the same color.
        for i in range(4):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[box_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/4, 
                language_goal=self.lang_template.format(color=colors[i]))