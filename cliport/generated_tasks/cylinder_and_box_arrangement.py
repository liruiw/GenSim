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

class CylinderAndBoxArrangement(Task):
    """Pick up each cylinder and place it inside the box of the same color, while avoiding collision with other objects."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "put the {color} cylinder in the {color} box"
        self.task_completed_desc = "done arranging cylinders and boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Cylinder and box colors.
        colors = ['red', 'blue', 'yellow']

        # Add cylinders and boxes.
        cylinder_size = (0.04, 0.04, 0.04)
        box_size = (0.12, 0.12, 0.12)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        box_urdf = 'box/box-template.urdf'

        cylinders = []
        boxes = []
        for color in colors:
            # Add cylinder
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=color)
            cylinders.append(cylinder_id)

            # Add box
            box_pose = self.get_random_pose(env, box_size)
            box_id = env.add_object(box_urdf, box_pose, color=color)
            boxes.append(box_id)

        # Add distractor blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block-template.urdf'
        for _ in range(5):
            block_pose = self.get_random_pose(env, block_size)
            env.add_object(block_urdf, block_pose)

        # Goal: each cylinder is in the box of the same color.
        for i in range(len(colors)):
            language_goal = self.lang_template.format(color=colors[i])
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[p.getBasePositionAndOrientation(boxes[i])], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/3, language_goal=language_goal)