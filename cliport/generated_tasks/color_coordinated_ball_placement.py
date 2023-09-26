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

class ColorCoordinatedBallPlacement(Task):
    """Pick up each ball and place it inside the box of the same color, while avoiding collision with other objects."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "put the {color} ball in the {color} box"
        self.task_completed_desc = "done placing balls in boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Ball and box colors.
        colors = ['red', 'blue', 'yellow']

        # Add balls and boxes.
        ball_size = (0.04, 0.04, 0.04)
        box_size = (0.12, 0.12, 0.12)
        ball_urdf = 'ball/ball-template.urdf'
        box_urdf = 'box/box-template.urdf'

        balls = []
        boxes = []
        for color in colors:
            ball_pose = self.get_random_pose(env, ball_size)
            box_pose = self.get_random_pose(env, box_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=utils.COLORS[color])
            box_id = env.add_object(box_urdf, box_pose, color=utils.COLORS[color])
            balls.append(ball_id)
            boxes.append(box_id)

        # Add distractor blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in colors]
        for _ in range(5):
            block_pose = self.get_random_pose(env, block_size)
            color = block_colors[np.random.randint(len(block_colors))]
            env.add_object(block_urdf, block_pose, color=color)

        # Goal: each ball is in the box of the same color.
        for i in range(len(balls)):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(boxes[i])], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / len(balls), language_goal=language_goal)