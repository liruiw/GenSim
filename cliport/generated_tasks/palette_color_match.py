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

class PaletteColorMatch(Task):
    """Pick up colored balls and place them into the corresponding colored slots in the pallet in a specific sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the {color} ball in the {color} slot"
        self.task_completed_desc = "done placing balls in slots."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.3, 0.3, 0.05)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, category='fixed')

        # Ball colors.
        colors = ['blue', 'yellow', 'green', 'red']

        # Add balls.
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball.urdf'

        objs = []
        for i in range(4):
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=utils.COLORS[colors[i]])
            objs.append(ball_id)

        # IMPORTANT Associate placement locations for goals.
        place_pos = [(0, -0.1, 0.03), (0, 0, 0.03),
                     (0, 0.1, 0.03), (0, 0.2, 0.03)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: balls are placed in the pallet in a specific sequence.
        for i in range(4):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[objs[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2]*1, language_goal=language_goal)