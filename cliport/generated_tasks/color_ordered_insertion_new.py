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

class ColorOrderedInsertionNew(Task):
    """Insert differently-colored ell objects into the matching color fixture in a specific order."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "put the {color} L shape block in the L shape hole"
        self.task_completed_desc = "done with insertion."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their order
        colors = ['red', 'blue', 'green', 'yellow']
        color_order = {color: i for i, color in enumerate(colors)}

        # Add fixtures.
        fixture_size = (0.12, 0.12, 0.02)
        fixture_urdf = 'insertion/fixture.urdf'
        fixtures = []
        for color in colors:
            fixture_pose = self.get_random_pose(env, fixture_size)
            fixture_id = env.add_object(fixture_urdf, fixture_pose, color=utils.COLORS[color], category='fixed')
            fixtures.append(fixture_id)

        # Add ell objects.
        ell_size = (0.04, 0.04, 0.04)
        ell_urdf = 'insertion/ell.urdf'
        ells = []
        for color in colors:
            ell_pose = self.get_random_pose(env, ell_size)
            ell_id = env.add_object(ell_urdf, ell_pose, color=utils.COLORS[color])
            ells.append(ell_id)

        # Goal: each ell is inserted into the matching color fixture in the correct order.
        for i, ell in enumerate(ells):
            self.add_goal(objs=[ell], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(fixtures[i])], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(ells),
                          language_goal=self.lang_template.format(color=colors[i]))