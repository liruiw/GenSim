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

class ColorCoordinatedBlockInsertion(Task):
    """Insert colored blocks into corresponding colored fixtures while avoiding colored spheres."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert the {color} block into the {color} fixture"
        self.task_completed_desc = "done inserting blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors
        colors = ['red', 'blue', 'green', 'yellow']
        color_items = {}

        # Add blocks and fixtures
        for color in colors:
            # Add block
            block_size = (0.04, 0.04, 0.04)
            block_pose = self.get_random_pose(env, block_size)
            block_urdf = 'block/block.urdf'
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            
            # Add fixture
            fixture_size = (0.06, 0.06, 0.02)
            fixture_pose = self.get_random_pose(env, fixture_size)
            fixture_urdf = 'insertion/fixture.urdf'
            fixture_id = env.add_object(fixture_urdf, fixture_pose, color=utils.COLORS[color])

            color_items[color] = (block_id, fixture_id)

        # Add spheres
        sphere_size = (0.04, 0.04, 0.04)
        sphere_urdf = 'sphere/sphere.urdf'
        for color in colors:
            sphere_pose = self.get_random_pose(env, sphere_size)
            env.add_object(sphere_urdf, sphere_pose, color=utils.COLORS[color])

        # Add goals
        for color, (block_id, fixture_id) in color_items.items():
            language_goal = self.lang_template.format(color=color)
            self.add_goal(objs=[block_id], matches=np.ones((1, 1)), targ_poses=[fixture_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 4, language_goal=language_goal)