import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils

class BuildBridge(Task):
    """Construct a bridge using two yellow blocks and three blue blocks.
    Firstly, place the two yellow blocks on each of the two bases parallel to each other with a fair amount of space in between.
    Then, place the blue block horizontally on top of the yellow blocks."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "build a bridge using four yellow blocks and one long blue block"
        self.task_completed_desc = "done building bridge."

    def reset(self, env):
        super().reset(env)

        # Add yellow blocks.
        base_length = 0.04
        base_size = (base_length, base_length, base_length)
        base_block_urdf = "box/box-template.urdf"
        bridge_pose = ((0.5, 0.0, 0.0), (0, 0, 0, 1))  # fixed pose
        self.add_corner_anchor_for_pose(env, bridge_pose)

        base_block_urdf = self.fill_template(base_block_urdf,  {'DIM': base_size})
        anchor_base_poses = [(utils.apply(bridge_pose, (- 3 * base_length / 2,  0, 0.001)), bridge_pose[1]),
                        (utils.apply(bridge_pose, ( 3 * base_length / 2,  0, 0.001)), bridge_pose[1]),
                        (utils.apply(bridge_pose, (- 3 * base_length / 2,  0, 0.041)), bridge_pose[1]),
                        (utils.apply(bridge_pose, ( 3 * base_length / 2, 0, 0.041)), bridge_pose[1])]
        base_blocks = []

        for idx in range(4):
            base_block_pose = self.get_random_pose(env, base_size)
            base_block_id = env.add_object(base_block_urdf, base_block_pose, color=utils.COLORS['yellow'])
            base_blocks.append(base_block_id)

        # Add car body block.
        body_size = (0.12, 0.04, 0.02)  # x, y, z dimensions for the asset size
        body_block_urdf = "box/box-template.urdf"
        body_block_urdf = self.fill_template(body_block_urdf,  {'DIM': body_size})
        body_block_pose = self.get_random_pose(env, body_size)
        body_block_id = env.add_object(body_block_urdf, body_block_pose, color=utils.COLORS['blue'])
        anchor_body_poses = [bridge_pose]

        # Goal: Firstly, create the base of the car by positioning two red blocks side by side.
        self.add_goal(objs=base_blocks[:2],
                      matches=np.ones((2, 2)),
                      targ_poses=anchor_base_poses,
                      replace=False,
                      rotations=True,
                      metric='pose',
                      params=None,
                      step_max_reward=1./4,
                      language_goal="Firstly, place the two yellow blocks on each of the two bases parallel to each other with a fair amount of space in between.")

        self.add_goal(objs=base_blocks[2:],
                      matches=np.ones((2, 2)),
                      targ_poses=anchor_base_poses,
                      replace=False,
                      rotations=True,
                      metric='pose',
                      params=None,
                      step_max_reward=1./2,
                      language_goal="Place the two yellow blocks on each of the two bases parallel to each other with a fair amount of space in between.")

        # Then, add the car body by stacking a blue block on top of the base.
        self.add_goal(objs=[body_block_id],
                      matches=np.ones((1, 1)),
                      targ_poses=anchor_body_poses,
                      replace=False,
                      rotations=True,
                      metric='pose',
                      params=None,
                      step_max_reward=1./4,
                      language_goal="Then, place the blue block horizontally on top of the yellow blocks.")