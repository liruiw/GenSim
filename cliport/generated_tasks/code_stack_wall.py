import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os
import copy

class CodeStackWall(Task):
    """Arrange a cylinder in a zone marked by a green box on the tabletop."""
    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Build a wall by stacking three symmetric blocks on top of each other. The blocks should be placed in a row, with each block touching the previous and next one."
        self.task_completed_desc = "done building the wall."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.15, 0.15, 0.01)
        base_urdf = 'box/box-template.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, category='fixed')

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_color = utils.COLORS['red']
        blocks = []
        for _ in range(3):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_color)
            blocks.append(block_id)

        # Calculate target poses.
        target_poses = []
        for i in range(3):
            target_pose = ((0.5, 0.0, 0.005), (0, 0, 0, 1))
            target_poses.append(target_pose)

        # Add goals.
        for i in range(3):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[target_poses[i]], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 3,
                          language_goal=self.lang_template)