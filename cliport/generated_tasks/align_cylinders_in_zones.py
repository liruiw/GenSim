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

class AlignCylindersInZones(Task):
    """Place four differently colored cylinders each into a matching colored zone. 
    The zones are surrounded by small blocks, which the robot needs to move out of the way 
    without knocking them out of their respective zones."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} cylinder in the {color} zone"
        self.task_completed_desc = "done aligning cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Cylinder colors.
        colors = ['red', 'blue', 'green', 'yellow']

        # Add cylinders.
        # x, y, z dimensions for the asset size
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'

        cylinders = []
        for i in range(4):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[colors[i]])
            cylinders.append(cylinder_id)

        # Add zones.
        # x, y, z dimensions for the asset size
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'

        zones = []
        for i in range(4):
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(zone_urdf, zone_pose, color=utils.COLORS[colors[i]], category='fixed')
            zones.append(zone_pose)

        # Add small blocks around the zones.
        # x, y, z dimensions for the asset size
        block_size = (0.02, 0.02, 0.02)
        block_urdf = 'block/small.urdf'

        for _ in range(16):
            block_pose = self.get_random_pose(env, block_size)
            env.add_object(block_urdf, block_pose)

        # Goal: each cylinder is in a matching colored zone.
        for i in range(4):
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[zones[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/4,
                    language_goal=self.lang_template.format(color=colors[i]))