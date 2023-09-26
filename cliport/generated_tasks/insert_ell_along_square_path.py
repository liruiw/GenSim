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

class InsertEllAlongSquarePath(Task):
    """Pick up each ell block and insert it into the fixture of the same color. However, the robot must move each ell block along the marked square path to reach the fixture."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "move the {color} ell block into the {color} fixture"
        self.task_completed_desc = "done inserting ell blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Ell block colors.
        colors = ['red', 'blue', 'green', 'yellow']

        # Add ell blocks and fixtures.
        ell_size = (0.04, 0.04, 0.04)
        ell_urdf = 'insertion/ell.urdf'
        fixture_urdf = 'insertion/fixture.urdf'
        ell_blocks = []
        fixtures = []
        for color in colors:
            # Add ell block
            ell_pose = self.get_random_pose(env, ell_size)
            ell_id = env.add_object(ell_urdf, ell_pose, color=utils.COLORS[color])
            ell_blocks.append(ell_id)

            # Add fixture
            fixture_pose = self.get_random_pose(env, ell_size)
            fixture_id = env.add_object(fixture_urdf, fixture_pose, color=utils.COLORS[color])
            fixtures.append(fixture_id)

        # Goal: each ell block is inserted into the fixture of the same color.
        for i in range(len(colors)):
            self.add_goal(objs=[ell_blocks[i]], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(fixtures[i])], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/len(colors),
                          language_goal=self.lang_template.format(color=colors[i]))

        # Add square path marked by small blocks.
        path_block_size = (0.02, 0.02, 0.02)
        path_block_urdf = 'block/small.urdf'
        path_block_color = utils.COLORS['gray']
        for _ in range(16):
            path_block_pose = self.get_random_pose(env, path_block_size)
            env.add_object(path_block_urdf, path_block_pose, color=path_block_color)