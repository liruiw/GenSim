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

class ColorCoordinatedBallInsertion(Task):
    """Insert balls into the cylinders of the same color in the order of red, blue, green, and yellow."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "insert the {color} ball into the {color} cylinder"
        self.task_completed_desc = "done inserting balls into cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their order
        colors = ['red', 'blue', 'green', 'yellow']

        # Add cylinders.
        # x, y, z dimensions for the asset size
        cylinder_size = (0.05, 0.05, 0.1)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinder_poses = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            env.add_object(cylinder_urdf, cylinder_pose, category='fixed', color=utils.COLORS[colors[i]])
            cylinder_poses.append(cylinder_pose)

        # Add balls.
        # x, y, z dimensions for the asset size
        balls = []
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        for i in range(4):
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=utils.COLORS[colors[i]])
            balls.append(ball_id)

        # Goal: each ball is in the corresponding color cylinder.
        for i in range(4):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[cylinder_poses[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/4,
                    language_goal=self.lang_template.format(color=colors[i]))