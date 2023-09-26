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

class ColorCoordinatedSphereOnPalletPyramid(Task):
    """Build a pyramid of colored blocks on pallets and place a matching colored sphere on top."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "build a pyramid of {color} blocks on the pallet and place the {color} sphere on top"
        self.task_completed_desc = "done building color-coordinated pyramids."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Pallets and Blocks
        pallet_size = (0.15, 0.15, 0.01)
        block_size = (0.04, 0.04, 0.04)
        pallet_urdf = 'pallet/pallet.urdf'
        block_urdf = 'block/block.urdf'

        # Colors for blocks and spheres
        colors = ['red', 'blue', 'green']
        color_objects = {}

        # Add pallets and blocks
        for color in colors:
            # Add pallet
            pallet_pose = self.get_random_pose(env, pallet_size)
            env.add_object(pallet_urdf, pallet_pose, category='fixed')

            # Add blocks
            block_ids = []
            for _ in range(3):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
                block_ids.append(block_id)

            color_objects[color] = {'pallet': pallet_pose, 'blocks': block_ids}

        # Spheres
        sphere_size = (0.04, 0.04, 0.04)
        sphere_urdf = 'sphere/sphere.urdf'

        # Add spheres
        for color in colors:
            sphere_pose = self.get_random_pose(env, sphere_size)
            sphere_id = env.add_object(sphere_urdf, sphere_pose, color=utils.COLORS[color])
            color_objects[color]['sphere'] = sphere_id

        # Goals
        for color in colors:
            # Goal: blocks are stacked in a pyramid on the pallet
            block_poses = [(0, -0.02, 0.02), (0, 0.02, 0.02), (0, 0, 0.06)]
            targs = [(utils.apply(color_objects[color]['pallet'], i), color_objects[color]['pallet'][1]) for i in block_poses]

            self.add_goal(objs=color_objects[color]['blocks'], matches=np.ones((3, 3)), targ_poses=targs, replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2]*3,
                    language_goal=self.lang_template.format(color=color))

            # Goal: sphere is placed on top of the pyramid
            sphere_pose = (0, 0, 0.1)
            targ = (utils.apply(color_objects[color]['pallet'], sphere_pose), color_objects[color]['pallet'][1])

            self.add_goal(objs=[color_objects[color]['sphere']], matches=np.ones((1, 1)), targ_poses=[targ], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 2, symmetries=[np.pi/2],
                          language_goal=self.lang_template.format(color=color))