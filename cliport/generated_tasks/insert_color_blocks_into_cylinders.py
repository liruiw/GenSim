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

class InsertColorBlocksIntoCylinders(Task):
    """Insert colored blocks into matching colored cylinders in a specific sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "insert the {color} block into the {color} cylinder"
        self.task_completed_desc = "done inserting blocks into cylinders."
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add blocks and cylinders.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        cylinder_size = (0.05, 0.05, 0.1)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'

        blocks = []
        cylinders = []
        for color in self.colors:
            # Add block
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

            # Add cylinder
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS[color])
            cylinders.append(cylinder_id)

        # Goal: each block is in the cylinder of the same color.
        for i, color in enumerate(self.colors):
            language_goal = self.lang_template.format(color=color)
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[cylinder_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(self.colors),
                          language_goal=language_goal)