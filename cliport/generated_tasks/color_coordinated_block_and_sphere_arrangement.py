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

class ColorCoordinatedBlockAndSphereArrangement(Task):
    """Arrange each colored block on top of the matching colored sphere while avoiding collisions with other scattered smaller blocks."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {} block on the {} sphere"
        self.task_completed_desc = "done arranging blocks and spheres."
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add blocks and spheres.
        block_size = (0.04, 0.04, 0.04)
        sphere_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        sphere_urdf = 'sphere/sphere-template.urdf'
        blocks = []
        spheres = []

        for color in self.colors:
            block_pose = self.get_random_pose(env, block_size)
            sphere_pose = self.get_random_pose(env, sphere_size)
            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            sphere_id = env.add_object(sphere_urdf, sphere_pose, color=utils.COLORS[color])
            blocks.append(block_id)
            spheres.append(sphere_id)

        # Add smaller blocks as distractors.
        small_block_urdf = 'block/small.urdf'
        for _ in range(5):
            small_block_pose = self.get_random_pose(env, block_size)
            env.add_object(small_block_urdf, small_block_pose)

        # Goal: each block is on top of the matching colored sphere.
        for i in range(len(blocks)):
            language_goal = self.lang_template.format(self.colors[i], self.colors[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[sphere_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(blocks),
                          language_goal=language_goal)