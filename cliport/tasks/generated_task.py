import numpy as np
from cliport.tasks import primitives
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os

class GeneratedTask(Task):
    """Construct a pyramid with six blocks of different colors (red, blue, green, yellow, orange, and purple). Arrange the blocks in a specific order to form a pyramid on the tabletop."""
    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the block on top of the cylinder on the pallet"
        self.task_completed_desc = "done placing block on cylinder."
        self.additional_reset()
    def reset(self, env):
        super().reset(env)
        # Add pallet.
        pallet_size = (0.35, 0.35, 0.01)
        pallet_pose = self.get_random_pose(env, pallet_size)
        pallet_urdf = 'pallet/pallet.urdf'
        env.add_object(pallet_urdf, pallet_pose, 'fixed')
        # Add cylinder.
        cylinder_size = (0.05, 0.05, 0.1)
        cylinder_pose = self.get_random_pose(env, cylinder_size)
        cylinder_urdf = 'cylinder/cylinder.urdf'
        cylinder_id = env.add_object(cylinder_urdf, cylinder_pose)
        # Add block.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        block_pose = self.get_random_pose(env, block_size)
        block_id = env.add_object(block_urdf, block_pose)
        # Goal: the block is on top of the cylinder on the pallet.
        self.add_goal(objs=[block_id], matches=np.ones((1, 1)), targ_poses=[cylinder_pose], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1,
                language_goal=self.lang_template)