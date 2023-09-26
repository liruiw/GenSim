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

class NestedBoxes(Task):
    """Place smaller boxes of different colors inside a larger box in a specific order."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "place the {color} box inside the yellow box"
        self.task_completed_desc = "done nesting boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add larger box.
        large_box_size = (0.12, 0.12, 0.12)
        large_box_pose = self.get_random_pose(env, large_box_size)
        large_box_urdf = 'box/box-template.urdf'
        large_box_id = env.add_object(large_box_urdf, large_box_pose, 'fixed')

        # Add smaller boxes.
        small_box_size = (0.04, 0.04, 0.04)
        small_box_urdf = 'box/box-template.urdf'
        colors = ['blue', 'red', 'green']
        boxes = []
        for color in colors:
            small_box_pose = self.get_random_pose(env, small_box_size)
            small_box_id = env.add_object(small_box_urdf, small_box_pose, color=color)
            boxes.append(small_box_id)

        # Goal: each small box is inside the larger box.
        for i, box in enumerate(boxes):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[box], matches=np.ones((1, 1)), targ_poses=[large_box_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/3, language_goal=language_goal)