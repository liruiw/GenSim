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

class BallInBowlObstacleCourseNew(Task):
    """Navigate through a maze of blocks, pick up balls of different colors and place them in the corresponding colored bowls."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {color} ball in the {color} bowl"
        self.task_completed_desc = "done placing balls in bowls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add blocks to form a maze.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/small.urdf'
        for _ in range(10):
            block_pose = self.get_random_pose(env, block_size)
            env.add_object(block_urdf, block_pose, category='fixed')

        # Add balls of different colors.
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        colors = ['red', 'blue', 'green', 'yellow']
        balls = []
        for color in colors:
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=color)
            balls.append(ball_id)

        # Add bowls of different colors at different corners of the maze.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowls = []
        for color in colors:
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, color=color, category='fixed')
            bowls.append(bowl_id)

        # Goal: each ball is in the bowl of the same color.
        for i in range(len(balls)):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(bowls[i])], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/len(balls),
                    language_goal=self.lang_template.format(color=colors[i]))