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

class ColorCoordinatedBlockCornerAlignment(Task):
    """Align each colored block to the corner of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 4
        self.lang_template = "place the {} block in the {} corner"
        self.task_completed_desc = "done aligning blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block and corner colors
        colors = ['red', 'green', 'blue', 'yellow']
        color_names = ['red', 'green', 'blue', 'yellow']

        # Add corners
        corner_size = (0.05, 0.05, 0.05)
        corner_urdf = 'corner/corner-template.urdf'
        corner_poses = []
        for i in range(4):
            corner_pose = self.get_random_pose(env, corner_size)
            env.add_object(corner_urdf, corner_pose, 'fixed', color=utils.COLORS[colors[i]])
            corner_poses.append(corner_pose)

        # Add blocks
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[colors[i]])
            blocks.append(block_id)

        # Add goals
        for i in range(4):
            language_goal = self.lang_template.format(color_names[i], color_names[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[corner_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 4, language_goal=language_goal)