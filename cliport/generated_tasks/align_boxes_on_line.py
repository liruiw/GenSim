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

class AlignBoxesOnLine(Task):
    """Align boxes of different colors on a single green line in a specific color sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "align the {color} box on the green line"
        self.task_completed_desc = "done aligning boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add green line.
        line_size = (0.3, 0.01, 0.01)
        line_pose = self.get_random_pose(env, line_size)
        env.add_object('line/single-green-line-template.urdf', line_pose, 'fixed')

        # Box colors.
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green']]

        # Add boxes.
        box_size = (0.04, 0.04, 0.04)
        box_urdf = 'box/box-template.urdf'
        boxes = []
        for i in range(3):
            box_pose = self.get_random_pose(env, box_size)
            box_id = env.add_object(box_urdf, box_pose, color=colors[i])
            boxes.append(box_id)

        # Goal: each box is aligned with the green line in a specific color sequence.
        # IMPORTANT Associate placement locations for goals.
        place_pos = [(-0.1, 0, 0.02), (0, 0, 0.02), (0.1, 0, 0.02)]
        targs = [(utils.apply(line_pose, i), line_pose[1]) for i in place_pos]

        # Add goals
        for i in range(3):
            color_name = ['red', 'blue', 'green'][i]
            language_goal = self.lang_template.format(color=color_name)
            self.add_goal(objs=[boxes[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/3, language_goal=language_goal)