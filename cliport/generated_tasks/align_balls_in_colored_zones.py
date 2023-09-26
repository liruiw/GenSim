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

class AlignBallsInColoredZones(Task):
    """Align balls of different colors in correspondingly colored zones."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} ball in the {color} zone"
        self.task_completed_desc = "done aligning balls in colored zones."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for balls and zones
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
        color_names = ['Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple']

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        for i in range(6):
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[colors[i]])
            zone_poses.append(zone_pose)

        # Add balls.
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        balls = []
        for i in range(6):
            ball_pose = self.get_random_pose(env, ball_size)
            replace = {'DIM': ball_size, 'HALF': (ball_size[0] / 2, ball_size[1] / 2, ball_size[2] / 2), 'COLOR': colors[i]}
            ball_urdf = self.fill_template(ball_urdf, replace)
            ball_id = env.add_object(ball_urdf, ball_pose)
            balls.append(ball_id)

        # Goal: each ball is in a different colored zone.
        for i in range(6):
            self.add_goal(objs=[balls[i]], matches=np.int32([[1]]), targ_poses=[zone_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 6,
                language_goal=self.lang_template.format(color=color_names[i]))