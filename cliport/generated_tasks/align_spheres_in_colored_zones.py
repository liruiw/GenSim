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

class AlignSpheresInColoredZones(Task):
    """Align spheres of different colors in the matching colored zones."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} sphere in the {color} zone"
        self.task_completed_desc = "done aligning spheres in colored zones."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors
        colors = ['red', 'blue', 'green', 'yellow']
        color_names = ['red', 'blue', 'green', 'yellow']

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone_poses = []
        for color in colors:
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, 'fixed', color=utils.COLORS[color])
            zone_poses.append(zone_pose)

        # Add spheres.
        sphere_size = (0.04, 0.04, 0.04)
        sphere_urdf = 'sphere/sphere-template.urdf'
        spheres = []
        for i, color in enumerate(colors):
            sphere_pose = self.get_random_pose(env, sphere_size)
            replace = {'DIM': sphere_size, 'HALF': (sphere_size[0] / 2, sphere_size[1] / 2, sphere_size[2] / 2)}
            sphere_urdf = self.fill_template(sphere_urdf, replace)
            sphere_id = env.add_object(sphere_urdf, sphere_pose, color=utils.COLORS[color])
            spheres.append(sphere_id)

            # Add goal
            self.add_goal(objs=[sphere_id], matches=np.ones((1, 1)), targ_poses=[zone_poses[i]], replace=False,
                          rotations=False, metric='pose', params=None, step_max_reward=1,
                          language_goal=self.lang_template.format(color=color_names[i]))