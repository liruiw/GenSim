import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os
import copy

class CodeBlueOnRedCylinder(Task):
    """Place a blue box on top of a red cylinder on a tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the blue box on top of the red cylinder"
        self.task_completed_desc = "done placing blue box on red cylinder."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add cylinder.
        cylinder_size = (0.05, 0.05, 0.1)
        cylinder_pose = self.get_random_pose(env, cylinder_size)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, 'fixed')

        # Add box.
        box_size = (0.04, 0.04, 0.04)
        box_pose = self.get_random_pose(env, box_size)
        box_urdf = 'box/box-template.urdf'
        box_id = env.add_object(box_urdf, box_pose, color=utils.COLORS['blue'])

        # Goal: the blue box is on top of the red cylinder.
        self.add_goal(objs=[box_id], matches=np.ones((1, 1)), targ_poses=[cylinder_pose], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1,
                language_goal=self.lang_template)