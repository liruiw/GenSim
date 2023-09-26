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

class ColorCoordinatedCylinderStandAssembly(Task):
    """Pick up each cylinder and place it on top of the stand of the same color, in a specific color sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the {color} cylinder on the {color} stand"
        self.task_completed_desc = "done placing cylinders on stands."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their sequence
        colors = ['green', 'yellow', 'blue', 'red']
        color_sequence = [utils.COLORS[color] for color in colors]

        # Add stands.
        stand_size = (0.04, 0.04, 0.04)
        stand_urdf = 'stacking/stand.urdf'
        stand_poses = []
        for i in range(4):
            stand_pose = self.get_random_pose(env, stand_size)
            env.add_object(stand_urdf, stand_pose, color=color_sequence[i], category='fixed')
            stand_poses.append(stand_pose)

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=color_sequence[i])
            cylinders.append(cylinder_id)

        # Goal: each cylinder is on the stand of the same color, in the specified color sequence.
        for i in range(4):
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[stand_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4, 
                          language_goal=self.lang_template.format(color=colors[i]))