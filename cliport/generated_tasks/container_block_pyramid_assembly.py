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

class ContainerBlockPyramidAssembly(Task):
    """Build a pyramid of colored blocks in a color sequence in matching containers"""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "put the {blocks} blocks in the {color} container and stack them in a pyramid"
        self.task_completed_desc = "done stacking block pyramid in container."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Block colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow']
        ]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'

        # Add containers.
        container_size = (0.12, 0.12, 0.12)
        container_urdf = 'container/container-template.urdf'

        objs = []
        for i in range(4):
            # Add 3 blocks of each color
            for _ in range(3):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=colors[i])
                objs.append(block_id)

            # Add container of matching color
            container_pose = self.get_random_pose(env, container_size)
            env.add_object(container_urdf, container_pose, color=colors[i], category='fixed')

        # IMPORTANT Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(container_pose, i), container_pose[1]) for i in place_pos]

        # Goal: blocks are stacked in a pyramid in the matching color container.
        for i in range(4):
            language_goal = self.lang_template.format(blocks="three", color=colors[i])
            self.add_goal(objs=objs[i*3:(i+1)*3], matches=np.ones((3, 3)), targ_poses=targs[:3], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2]*3, language_goal=language_goal)