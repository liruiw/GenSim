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

class StackBoxesInPallet(Task):
    """Stack boxes in a pallet in a specific color sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "stack the {color} box on the pallet"
        self.task_completed_desc = "done stacking boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.15, 0.15, 0.02)
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object('pallet/pallet.urdf', pallet_pose, 'fixed')

        # Box colors.
        colors = [utils.COLORS['yellow'], utils.COLORS['green'], utils.COLORS['blue']]
        color_names = ['yellow', 'green', 'blue']

        # Add boxes.
        box_size = (0.04, 0.04, 0.04)
        box_urdf = 'box/box-template.urdf'
        boxes = []
        for i in range(3):
            box_pose = self.get_random_pose(env, box_size)
            box_id = env.add_object(box_urdf, box_pose, color=colors[i])
            boxes.append(box_id)

        # Goal: boxes are stacked on the pallet in a specific color sequence.
        for i in range(3):
            language_goal = self.lang_template.format(color=color_names[i])
            self.add_goal(objs=[boxes[i]], matches=np.ones((1, 1)), targ_poses=[pallet_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/3, language_goal=language_goal)