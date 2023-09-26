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

class ColorCoordinatedCylinderPyramid(Task):
    """Construct a pyramid on a pallet using four cylinders of different colors (red, blue, green, and yellow)."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "make the {row} row with {cylinder}"
        self.task_completed_desc = "done stacking cylinder pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.15, 0.15, 0.005)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, category='fixed')

        # Cylinder colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow']
        ]

        # Add cylinders.
        # x, y, z dimensions for the asset size
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'

        cylinders = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=colors[i])
            cylinders.append(cylinder_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0.05, 0.03),
                     (0, 0, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: cylinders are stacked in a pyramid (bottom row: red, blue).
        self.add_goal(objs=cylinders[:2], matches=np.ones((2, 2)), targ_poses=targs[:2], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*2,
                language_goal=self.lang_template.format(cylinder="the red and blue cylinders", row="bottom"))

        # Goal: cylinders are stacked in a pyramid (middle row: green).
        self.add_goal(objs=cylinders[2:3], matches=np.ones((1, 1)), targ_poses=targs[2:3], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2]*1,
                language_goal=self.lang_template.format(cylinder="the green cylinder", row="middle"))

        # Goal: cylinders are stacked in a pyramid (top row: yellow).
        self.add_goal(objs=cylinders[3:], matches=np.ones((1, 1)), targ_poses=targs[3:], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2]*1,
                language_goal=self.lang_template.format(cylinder="the yellow cylinder", row="top"))