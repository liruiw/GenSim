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

class ColorCoordinatedBoxInPallet(Task):
    """Pick up boxes of different colors and place them into the pallet in a specific color sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the {color} box into the pallet"
        self.task_completed_desc = "done placing boxes in the pallet."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.15, 0.15, 0.05)
        pallet_pose = self.get_random_pose(env, pallet_size)
        pallet_urdf = 'pallet/pallet.urdf'
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Define box colors and sizes.
        box_colors = ['red', 'blue', 'green', 'yellow']
        box_size = (0.04, 0.04, 0.04)
        box_urdf = 'box/box-template.urdf'

        # Add boxes.
        boxes = []
        for color in box_colors:
            box_pose = self.get_random_pose(env, box_size)
            box_id = env.add_object(box_urdf, box_pose, color=color)
            boxes.append(box_id)

        # Goal: each box is in the pallet in a specific color sequence.
        for i, box in enumerate(boxes):
            language_goal = self.lang_template.format(color=box_colors[i])
            self.add_goal(objs=[box], matches=np.ones((1, 1)), targ_poses=[pallet_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(boxes),
                          language_goal=language_goal)