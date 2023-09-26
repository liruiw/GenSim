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

class ColorCuedBallCornerSorting(Task):
    """Pick up each colored ball and place it in the corner of the same color while avoiding a zone marked by small blocks."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} ball in the {color} corner"
        self.task_completed_desc = "done sorting balls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add corners.
        corner_size = (0.05, 0.05, 0.05)
        corner_urdf = 'corner/corner-template.urdf'
        corner_colors = ['red', 'blue', 'green', 'yellow']
        corner_poses = []
        for color in corner_colors:
            corner_pose = self.get_random_pose(env, corner_size)
            env.add_object(corner_urdf, corner_pose, color=color, category='fixed')
            corner_poses.append(corner_pose)

        # Add balls.
        balls = []
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        for color in corner_colors:
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=color)
            balls.append(ball_id)

        # Add zone.
        zone_size = (0.2, 0.2, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        zone_urdf = 'zone/zone.urdf'
        env.add_object(zone_urdf, zone_pose, 'fixed')

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block_for_anchors.urdf'
        for _ in range(4):
            block_pose = self.get_random_pose(env, block_size)
            env.add_object(block_urdf, block_pose)

        # Goal: each ball is in the corner of the same color.
        for i in range(4):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[corner_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4,
                          language_goal=self.lang_template.format(color=corner_colors[i]))