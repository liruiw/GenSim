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

class BuildCylinderStructure(Task):
    """Construct a structure using four colored cylinders (red, blue, green, yellow) on a square base."""

    def __init__(self):
        super().__init__()
        self.max_steps = 5
        self.lang_template = "construct a structure using four colored cylinders on a square base"
        self.task_completed_desc = "done building the cylinder structure."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add square base.
        # x, y, z dimensions for the asset size
        base_size = (0.15, 0.15, 0.005)
        base_urdf = 'square/square-template.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, category='fixed')

        # Cylinder colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']
        ]

        # Add cylinders.
        # x, y, z dimensions for the asset size
        cylinder_size = (0.04, 0.04, 0.08)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'

        objs = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=colors[i])
            objs.append(cylinder_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.04), (0, 0.05, 0.04),
                     (0, 0.05, 0.12), (0, -0.05, 0.12)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: red and blue cylinders are placed side by side on the base.
        self.add_goal(objs=objs[:2], matches=np.ones((2, 2)), targ_poses=targs[:2], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*2,
                language_goal="place the red and blue cylinders side by side on the base")

        # Goal: green cylinder is placed on top of the blue cylinder.
        self.add_goal(objs=[objs[2]], matches=np.ones((1, 1)), targ_poses=[targs[2]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2],
                language_goal="place the green cylinder on top of the blue cylinder")

        # Goal: yellow cylinder is placed on top of the red cylinder.
        self.add_goal(objs=[objs[3]], matches=np.ones((1, 1)), targ_poses=[targs[3]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2],
                language_goal="place the yellow cylinder on top of the red cylinder")