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

class CylinderStandAlignment(Task):
    """Arrange four colored cylinders (red, blue, green, yellow) in order of their colors on four stands of matching color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Arrange the {color} cylinder on the {color} stand"
        self.task_completed_desc = "done arranging cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']]
        color_names = ['red', 'blue', 'green', 'yellow']

        # Add cylinders.
        # x, y, z dimensions for the asset size
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            replace = {'DIM': cylinder_size, 'HALF': (cylinder_size[0] / 2, cylinder_size[1] / 2, cylinder_size[2] / 2),
                       'COLOR': colors[i]}
            # IMPORTANT: REPLACE THE TEMPLATE URDF
            urdf = self.fill_template(cylinder_urdf, replace)
            cylinder_id = env.add_object(urdf, cylinder_pose)
            cylinders.append(cylinder_id)

        # Add stands.
        # x, y, z dimensions for the asset size
        stand_size = (0.05, 0.05, 0.005)
        stand_urdf = 'stacking/stand.urdf'
        stands = []
        for i in range(4):
            stand_pose = self.get_random_pose(env, stand_size)
            env.add_object(stand_urdf, stand_pose, color=colors[i], category='fixed')
            stands.append(stand_pose)

        # Goal: each cylinder is on a stand of the same color.
        for i in range(4):
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[stands[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 4,
                          language_goal=self.lang_template.format(color=color_names[i]))