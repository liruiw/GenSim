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

class MixedColorBlockBarrierInsertion(Task):
    """Pick up each colored block, navigate the barriers, and insert each block into the fixture of the same color."""

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

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for color in colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Add fixtures.
        fixture_size = (0.06, 0.06, 0.06)
        fixture_urdf = 'insertion/fixture.urdf'
        fixtures = []
        for color in colors:
            fixture_pose = self.get_random_pose(env, fixture_size)
            fixture_id = env.add_object(fixture_urdf, fixture_pose, color=utils.COLORS[color])
            fixtures.append(fixture_id)

        # Add barriers.
        barrier_size = (0.12, 0.04, 0.04)
        barrier_colors = ['orange', 'purple', 'brown']
        for _ in range(2):
            for color in barrier_colors:
                barrier_pose = self.get_random_pose(env, barrier_size)
                env.add_object(block_urdf, barrier_pose, color=utils.COLORS[color])

        # Goal: each block is inserted into the fixture of the same color.
        for i in range(len(blocks)):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(fixtures[i])], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/len(blocks),
                          language_goal=self.lang_template.format(color=colors[i]))