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

class ColorCoordinatedCylinderTower(Task):
    """Stack cylinders of four different colors (red, blue, green, yellow) on top of each other on a square stand in a specific sequence. The bottom of the stack should start with a blue cylinder, follow by a green cylinder, then a red one, and finally a yellow cylinder at the top. Each cylinder has to be aligned correctly to avoid falling."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "Stack cylinders of four different colors (red, blue, green, yellow) on top of each other on a square stand in a specific sequence. The bottom of the stack should start with a blue cylinder, follow by a green cylinder, then a red one, and finally a yellow cylinder at the top."
        self.task_completed_desc = "done stacking cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, category='fixed')

        # Cylinder colors.
        colors = [utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['red'], utils.COLORS['yellow']]

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'

        objs = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=colors[i])
            objs.append(cylinder_id)

        # Associate placement locations for goals.
        place_pos = [(0, 0, 0.03), (0, 0, 0.08), (0, 0, 0.13), (0, 0, 0.18)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: cylinders are stacked in a tower (bottom to top: blue, green, red, yellow).
        for i in range(4):
            self.add_goal(objs=[objs[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2],
                          language_goal=self.lang_template)