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

class ColorfulBlockTowerOnCylinderBase(Task):
    """Construct a tower using four blocks of different colors (red, blue, green, and yellow) on a placed cylindrical base at the corner of the tabletop. The sequence from bottom to top should be red, blue, green, and yellow."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "construct a tower using four blocks of different colors (red, blue, green, and yellow) on a placed cylindrical base at the corner of the tabletop. The sequence from bottom to top should be red, blue, green, and yellow."
        self.task_completed_desc = "done building the tower."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add cylindrical base.
        # x, y, z dimensions for the asset size
        base_size = (0.05, 0.05, 0.05)
        base_urdf = 'cylinder/cylinder-template.urdf'
        base_pose = self.get_random_pose(env, base_size)
        base_id = env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow']]

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'

        objs = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            objs.append(block_id)

        # Associate placement locations for goals.
        place_pos = [(0, 0, 0.05), (0, 0, 0.09), (0, 0, 0.13), (0, 0, 0.17)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: blocks are stacked on the cylindrical base in the order red, blue, green, yellow from bottom to top.
        for i in range(4):
            self.add_goal(objs=[objs[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2],
                          language_goal=self.lang_template)