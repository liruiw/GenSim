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

class BallSortingWithBlocksBarrier(Task):
    """Pick up each ball and place it into the zone of the same color, but without knocking over the blocks."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} ball in the {color} zone without knocking over the blocks"
        self.task_completed_desc = "done sorting balls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for the balls and zones
        colors = ['red', 'blue', 'green', 'yellow']

        # Add zones and blocks.
        zone_size = (0.12, 0.12, 0)
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/small.urdf'
        zone_urdf = 'zone/zone.urdf'
        zones = []
        blocks = []
        for color in colors:
            # Add zone of specific color
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[color])
            zones.append(zone_pose)

            # Add line of blocks of the same color
            for _ in range(5):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
                blocks.append(block_id)

        # Add balls.
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        balls = []
        for color in colors:
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=utils.COLORS[color])
            balls.append(ball_id)

        # Goal: each ball is in a zone of the same color.
        for i in range(len(balls)):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[zones[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/len(balls),
                          language_goal=self.lang_template.format(color=colors[i]))