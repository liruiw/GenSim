import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import IPython

class ConnectBoxesWithRope(Task):
    """Connect two colored blocks with ropes."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "connect the {color1} and {color2} blocks with the rope."
        self.task_completed_desc = "done connecting."
        self.additional_reset()
        self.pos_eps = 0.04 # higher tolerance

    def reset(self, env):
        super().reset(env)
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        blocks = []
        target_colors = np.random.choice(colors, 2, replace=False)
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        corner_poses = []

        for color in colors:
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=color)
            blocks.append(block_id)
            if color in target_colors:
                corner_poses.append(block_pose)

        dist = np.linalg.norm(np.array(corner_poses[0][0])-np.array(corner_poses[1][0])) 
        n_parts = int(20 * dist / 0.4)

        # IMPORTANT: use `make_ropes` to add cable (series of articulated small blocks).
        objects, targets, matches = self.make_ropes(env, corners=(corner_poses[0][0], corner_poses[1][0]), n_parts=n_parts)
        self.add_goal(objs=objects, matches=matches, targ_poses=targets, replace=False,
                rotations=False, metric='pose', params=None, step_max_reward=1.,
                language_goal=self.lang_template.format(color1=target_colors[0], color2=target_colors[1]))

        # wait for the scene to settle down
        for i in range(600):
            p.stepSimulation()