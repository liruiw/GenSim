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
import pybullet as p

class ColorCoordinatedInsertion(Task):
    """Insert each block into the fixture of the same color"""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert each block into the fixture of the same color"
        self.task_completed_desc = "done with color-coordinated-insertion."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.35, 0.35, 0.01)
        pallet_pose = self.get_random_pose(env, pallet_size)
        pallet_urdf = 'pallet/pallet.urdf'
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Add fixtures and blocks.
        colors = ['red', 'blue', 'green', 'yellow']
        fixtures = []
        blocks = []
        fixture_size = (0.05, 0.05, 0.05)
        block_size = (0.04, 0.04, 0.04)
        fixture_urdf = 'insertion/fixture.urdf'
        block_urdf = 'block/block.urdf'
        for color in colors:
            # Add fixture.
            fixture_pose = self.get_random_pose(env, fixture_size)
            fixture_id = env.add_object(fixture_urdf, fixture_pose, color=utils.COLORS[color])
            fixtures.append(fixture_id)

            # Add block.
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Goal: each block is in the fixture of the same color.
        for i in range(len(blocks)):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(fixtures[i])], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(blocks),
                          language_goal=self.lang_template)

        # Goal: each fixture is on the pallet.
        for i in range(len(fixtures)):
            self.add_goal(objs=[fixtures[i]], matches=np.ones((1, 1)), targ_poses=[pallet_pose], replace=False,
                          rotations=True, metric='zone', params=[(pallet_pose, pallet_size)], step_max_reward=1 / len(fixtures),
                          language_goal=self.lang_template)