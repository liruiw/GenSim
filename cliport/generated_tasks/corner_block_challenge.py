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

class CornerBlockChallenge(Task):
    """Construct two columns using eight cubes - four red, two green, and two blue. 
    The columns should be constructed at two distinct marked corners of the tabletop 
    using the 'corner/corner-template.urdf' asset. The first column should be constructed 
    with the red cubes and the second column should use the green and blue cubes, 
    with blue at the base."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "construct two columns using eight cubes - four red, two green, and two blue"
        self.task_completed_desc = "done constructing columns."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add corners.
        corner_size = (0.15, 0.15, 0.01)
        corner_urdf = 'corner/corner-template.urdf'
        corner_poses = []
        for _ in range(2):
            corner_pose = self.get_random_pose(env, corner_size)
            env.add_object(corner_urdf, corner_pose, 'fixed')
            corner_poses.append(corner_pose)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['red']] * 4 + [utils.COLORS['green']] * 2 + [utils.COLORS['blue']] * 2
        blocks = []
        for i in range(8):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_colors[i])
            blocks.append(block_id)

        # Goal: each block is stacked in the correct corner in the correct order.
        for i in range(4):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[corner_poses[0]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/8,
                          language_goal=self.lang_template)

        for i in range(4, 8):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[corner_poses[1]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/8,
                          language_goal=self.lang_template)