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

class ColorCoordinatedCylinderStacking(Task):
    """Stack cylinders on a pallet in a specific color sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "stack the {color} cylinder on the pallet"
        self.task_completed_desc = "done stacking cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.15, 0.15, 0.01)
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object('pallet/pallet.urdf', pallet_pose, 'fixed')

        # Cylinder colors.
        colors = ['red', 'blue', 'green']

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.12)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(3):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            color = utils.COLORS[colors[i]]
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=color)
            cylinders.append(cylinder_id)

        # Goal: cylinders are stacked on the pallet in a specific color sequence.
        stack_poses = [(0, 0, 0.06), (0, 0, 0.12), (0, 0, 0.18)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in stack_poses]

        for i in range(3):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/3, language_goal=language_goal)