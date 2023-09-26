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

class MulticolorBlockBridge(Task):
    """Build a bridge by stacking three red, three blue, and three green blocks on a pallet. 
    Arrange in a sequence from left to right: red, blue, and green. 
    Then, place three cylinders of corresponding colors on top of the stacked blocks, forming a bridge. 
    The cylinders should roll from the top block to the pallet, creating a challenge of precision and control."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Build a bridge by stacking three red, three blue, and three green blocks on a pallet. Arrange in a sequence from left to right: red, blue, and green. Then, place three cylinders of corresponding colors on top of the stacked blocks, forming a bridge."
        self.task_completed_desc = "done building the bridge."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        # x, y, z dimensions for the asset size
        pallet_size = (0.15, 0.15, 0.01)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        # Add blocks.
        # x, y, z dimensions for the asset size
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green']]
        blocks = []
        for i in range(9):  # 3 blocks of each color
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=block_colors[i // 3])
            blocks.append(block_id)

        # Add cylinders.
        # x, y, z dimensions for the asset size
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_template = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(3):  # 1 cylinder of each color
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            replace = {'DIM': cylinder_size, 'HALF': (cylinder_size[0] / 2, cylinder_size[1] / 2, cylinder_size[2] / 2)}
            cylinder_urdf = self.fill_template(cylinder_template, replace)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=block_colors[i])
            cylinders.append(cylinder_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03), (0, 0.05, 0.03)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: blocks are stacked on the pallet in the order red, blue, green.
        for i in range(9):
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[targs[i // 3]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 9, symmetries=[np.pi/2],
                          language_goal=self.lang_template)

        # Goal: cylinders are placed on top of the stacked blocks.
        for i in range(3):
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 3, symmetries=[np.pi/2], 
                          language_goal=self.lang_template)