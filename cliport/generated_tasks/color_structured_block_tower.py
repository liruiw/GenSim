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

class ColorStructuredBlockTower(Task):
    """Construct a tower using six blocks: two red, two blue, and two green. 
    The tower should be built in the order of a red block at the base, 
    followed by a blue, then green, then red, blue and green at the top."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "construct a tower using six blocks: two red, two blue, and two green. " \
                             "The tower should be built in the order of a red block at the base, " \
                             "followed by a blue, then green, then red, blue and green at the top."
        self.task_completed_desc = "done building color-structured block tower."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block colors and sizes
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green']] * 2
        block_size = (0.04, 0.04, 0.04)

        # Add blocks
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i])
            blocks.append(block_id)

        # Define target poses for the blocks in the tower
        base_pose = self.get_random_pose(env, block_size)
        targ_poses = [base_pose]
        for i in range(1, 6):
            targ_poses.append((np.array(base_pose[0]) + np.array([0, 0, i * block_size[2]]), base_pose[1]))

        # Add goals
        for i in range(6):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[targ_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/6, symmetries=[np.pi/2],
                          language_goal=self.lang_template)