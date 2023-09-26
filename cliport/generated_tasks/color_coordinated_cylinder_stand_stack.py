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

class ColorCoordinatedCylinderStandStack(Task):
    """Arrange three cylinders of different colors (red, blue, green) on three stands of corresponding colors."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "place the {} cylinder on the {} stand"
        self.task_completed_desc = "done arranging cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green']]
        color_names = ['red', 'blue', 'green']

        # Add cylinders
        cylinder_size = (0.02, 0.02, 0.06)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(3):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=colors[i])
            cylinders.append(cylinder_id)

        # Add stands
        stand_size = (0.06, 0.06, 0.02)
        stand_urdf = 'stacking/stand.urdf'
        stands = []
        for i in range(3):
            stand_pose = self.get_random_pose(env, stand_size)
            stand_id = env.add_object(stand_urdf, stand_pose, color=colors[i])
            stands.append(stand_id)

        # Goal: each cylinder is on the stand of the same color
        for i in range(3):
            language_goal = self.lang_template.format(color_names[i], color_names[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[stand_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 3, language_goal=language_goal)