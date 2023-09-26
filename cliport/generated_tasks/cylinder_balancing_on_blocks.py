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

class CylinderBalancingOnBlocks(Task):
    """Construct a bridge using two red blocks placed parallel to each other with a gap in between, and then balance a green cylinder horizontally on top of the red blocks without it falling over."""

    def __init__(self):
        super().__init__()
        self.max_steps = 5
        self.lang_template = "balance the green cylinder on the red blocks"
        self.task_completed_desc = "done balancing."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_poses = []
        for _ in range(2):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS['red'])
            block_poses.append(block_pose)

        # Add cylinder.
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinder_pose = self.get_random_pose(env, cylinder_size)
        cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS['green'])

        # Goal: balance the cylinder on the blocks.
        self.add_goal(objs=[cylinder_id], matches=np.ones((1, 2)), targ_poses=block_poses, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1, language_goal=self.lang_template)