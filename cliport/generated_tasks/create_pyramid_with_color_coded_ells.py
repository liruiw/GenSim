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

class CreatePyramidWithColorCodedElls(Task):
    """Pick up ell-shaped objects of different colors and stack them onto a pallet in the shape of a pyramid."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "stack the {color} ell on the pyramid"
        self.task_completed_desc = "done stacking ell pyramid."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        pallet_size = (0.15, 0.15, 0.01)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = self.get_random_pose(env, pallet_size)
        env.add_object(pallet_urdf, pallet_pose, category='fixed')

        # Ell colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['blue'], 
            utils.COLORS['yellow'], utils.COLORS['green']
        ]
        color_names = ['red', 'blue', 'yellow', 'green']

        # Add Ells.
        ell_size = (0.04, 0.04, 0.04)
        ell_urdf = 'insertion/ell.urdf'
        objs = []
        for i in range(4):
            ell_pose = self.get_random_pose(env, ell_size)
            ell_id = env.add_object(ell_urdf, ell_pose, color=colors[i])
            objs.append(ell_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, 0, 0.08)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos]

        # Goal: Ells are stacked in a pyramid (bottom row: red, middle row: blue, top row: yellow, green).
        for i in range(4):
            self.add_goal(objs=[objs[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2],
                          language_goal=self.lang_template.format(color=color_names[i]))