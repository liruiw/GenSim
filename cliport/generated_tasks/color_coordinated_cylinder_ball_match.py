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

class ColorCoordinatedCylinderBallMatch(Task):
    """Pick up each ball and place it on top of the cylinder of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} ball on the {color} cylinder"
        self.task_completed_desc = "done placing balls on cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add cylinders.
        # x, y, z dimensions for the asset size
        cylinder_size = (0.04, 0.04, 0.1)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinder_colors = ['red', 'blue', 'green', 'yellow']
        cylinder_poses = []
        cylinders = []
        for color in cylinder_colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=color)
            cylinder_poses.append(cylinder_pose)
            cylinders.append(cylinder_id)

        # Add balls.
        # x, y, z dimensions for the asset size
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        balls = []
        for color in cylinder_colors:
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=color)
            balls.append(ball_id)

        # Add blocks as obstacles.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/small.urdf'
        for _ in range(5):
            block_pose = self.get_random_pose(env, block_size)
            env.add_object(block_urdf, block_pose)

        # Goal: each ball is on top of the cylinder of the same color.
        for i in range(len(balls)):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[cylinder_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/len(balls),
                language_goal=self.lang_template.format(color=cylinder_colors[i]))