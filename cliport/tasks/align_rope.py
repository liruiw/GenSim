import os

import numpy as np
from cliport.tasks import primitives
from cliport.tasks.task import Task
from cliport.utils import utils

import random
import pybullet as p


class AlignRope(Task):
    """Manipulate a deformable rope to connect its end-points between two 
    corners of a 3-sided square."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "align the rope from {direction}"
        self.task_completed_desc = "done aligning the rope."
        self.additional_reset()


    def reset(self, env):
        super().reset(env)

        n_parts = 20
        radius = 0.005
        length = 2 * radius * n_parts * np.sqrt(2)

        # Add 3-sided square.
        square_size = (length, length, 0)
        square_pose = self.get_random_pose(env, square_size)
        square_template = 'square/square-template.urdf'
        replace = {'DIM': (length,), 'HALF': (np.float32(length) / 2 - 0.005,)}

        # IMPORTANT: REPLACE THE TEMPLATE URDF
        urdf = self.fill_template(square_template, replace)
        env.add_object(urdf, square_pose, 'fixed')

        # Get four corner points of square.
        corner0 = ( length / 2,  length / 2, 0.001)
        corner1 = (-length / 2,  length / 2, 0.001)
        corner2 = ( length / 2, -length / 2, 0.001)
        corner3 = (-length / 2, -length / 2, 0.001)

        corner0 = utils.apply(square_pose, corner0)
        corner1 = utils.apply(square_pose, corner1)
        corner2 = utils.apply(square_pose, corner2)
        corner3 = utils.apply(square_pose, corner3)
        
        # Four possible alignment tasks.
        task_descs = [
            ((corner0, corner1), "front left tip to front right tip"),
            ((corner0, corner2), "front right tip to back right corner"),
            ((corner1, corner3), "front left tip to back left corner"),
            ((corner3, corner2), "back right corner to back left corner")
        ]
        chosen_task = np.random.choice(len(task_descs), 1)[0]
        (corner_0, corner_1), direction = task_descs[chosen_task]

        # IMPORTANT: use `make_ropes` to add cable (series of articulated small blocks).
        objects, targets, matches = self.make_ropes(env, corners=(corner_0, corner_1))
        self.add_goal(objs=objects, matches=matches, targ_poses=targets, replace=False,
                rotations=False, metric='pose', params=None, step_max_reward=1.,
                language_goal=[self.lang_template.format(direction=direction)] * len(objects))

        # wait for the scene to settle down
        for i in range(480):
            p.stepSimulation()