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

class AlignCylindersInSquare(Task):
    """Arrange four cylinders of different colors (red, blue, green, yellow) 
    on the corners of a square facing the center. The square is outlined by 
    four zones of matching colors at each corner. The red cylinder should be 
    placed at the red zone and facing the center of the square, and same for 
    the remaining colors."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "arrange the {color} cylinder at the {color} corner of the square"
        self.task_completed_desc = "done arranging cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding zones
        colors = ['red', 'blue', 'green', 'yellow']
        zones = ['zone-red', 'zone-blue', 'zone-green', 'zone-yellow']

        # Add zones at the corners of a square
        zone_size = (0.05, 0.05, 0.005)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        for i in range(4):
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed')
            zone_poses.append(zone_pose)

        # Add cylinders of different colors
        cylinder_size = (0.02, 0.02, 0.08)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            color = utils.COLORS[colors[i]]
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=color)
            cylinders.append(cylinder_id)

        # Goal: each cylinder is in a different zone of the same color
        for i in range(4):
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[zone_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4)
            self.lang_goals.append(self.lang_template.format(color=colors[i]))