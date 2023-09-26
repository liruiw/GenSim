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

class BuildWheel(Task):
    """Construct a wheel using blocks and a sphere."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "Construct a wheel using blocks and a sphere. First, position eight blocks in a circular layout on the tabletop. Each block should be touching its two neighbors and colored in alternating red and blue. Then place a green sphere in the center of the circular layout, completing the wheel."
        self.task_completed_desc = "done building wheel."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['red'], utils.COLORS['blue']]
        blocks = []
        for i in range(8):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_colors[i % 2])
            blocks.append(block_id)

        # Add sphere.
        sphere_size = (0.04, 0.04, 0.04)
        sphere_urdf = 'sphere/sphere.urdf'
        sphere_color = utils.COLORS['green']
        sphere_pose = ((0.5, 0.0, 0.0), (0,0,0,1)) # fixed pose
        sphere_id = env.add_object(sphere_urdf, sphere_pose, color=sphere_color)

        # Goal: blocks are arranged in a circle and sphere is in the center.
        circle_radius = 0.1
        circle_center = (0, 0, block_size[2] / 2)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        block_poses = [(circle_center[0] + circle_radius * np.cos(angle),
                        circle_center[1] + circle_radius * np.sin(angle),
                        circle_center[2]) for angle in angles]
        block_poses = [(utils.apply(sphere_pose, pos), sphere_pose[1]) for pos in block_poses]
        self.add_goal(objs=blocks, matches=np.ones((8, 8)), targ_poses=block_poses, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=8 / 9, language_goal=self.lang_template)

        # Goal: sphere is in the center of the blocks.
        self.add_goal(objs=[sphere_id], matches=np.ones((1, 1)), targ_poses=[sphere_pose], replace=False,
                rotations=False, metric='pose', params=None, step_max_reward=1 / 9, language_goal=self.lang_template)