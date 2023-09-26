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

class InsertBlocksLineup(Task):
    """Pick up four different color blocks and insert them into the corresponding color fixtures."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert the {color} block into the {color} fixture"
        self.task_completed_desc = "done inserting blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for blocks and fixtures
        colors = ['red', 'blue', 'green', 'yellow']

        # Add fixtures.
        fixture_size = (0.04, 0.04, 0.04)
        fixture_urdf = 'insertion/fixture.urdf'
        fixture_poses = []
        for i in range(4):
            fixture_pose = self.get_random_pose(env, fixture_size)
            fixture_id = env.add_object(fixture_urdf, fixture_pose, color=utils.COLORS[colors[i]], category='fixed')
            fixture_poses.append((fixture_pose, fixture_id))

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(4):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[colors[i]])
            blocks.append(block_id)

        # Add small blocks as barriers.
        small_block_size = (0.02, 0.02, 0.02)
        small_block_urdf = 'block/small.urdf'
        for _ in range(10):
            small_block_pose = self.get_random_pose(env, small_block_size)
            env.add_object(small_block_urdf, small_block_pose)

        # Goal: each block is in the corresponding color fixture.
        for i in range(4):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[fixture_poses[i][0]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/4,
                          language_goal=self.lang_template.format(color=colors[i]))