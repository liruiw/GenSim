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

class ColorCoordinatedLineBlocks(Task):
    """Move colored blocks along colored lines in a specific sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "move the {color} block to the other end of the {color} line"
        self.task_completed_desc = "done moving blocks."
        self.colors = ['red', 'blue', 'green', 'yellow', 'orange']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add lines.
        line_size = (0.3, 0.01, 0.01)
        line_urdf = 'line/line-template.urdf'
        line_poses = []
        for i in range(5):
            line_pose = self.get_random_pose(env, line_size)
            color = utils.COLORS[self.colors[i]]
            env.add_object(line_urdf, line_pose, 'fixed', color=color)
            line_poses.append(line_pose)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(5):
            block_pose = self.get_random_pose(env, block_size)
            color = utils.COLORS[self.colors[i]]
            block_id = env.add_object(block_urdf, block_pose, color=color)
            blocks.append(block_id)

        # Goal: each block is at the other end of its corresponding line.
        for i in range(5):
            language_goal = self.lang_template.format(color=self.colors[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[line_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/5, language_goal=language_goal)