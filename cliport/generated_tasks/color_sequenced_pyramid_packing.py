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

class ColorSequencedPyramidPacking(Task):
    """Sort cubes by color into four pallets and stack them in each pallet as a pyramid"""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "sort the {color} cubes into the pallet and stack them as a pyramid"
        self.task_completed_desc = "done sorting and stacking cubes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add pallets.
        # x, y, z dimensions for the asset size
        pallet_size = (0.15, 0.15, 0.02)
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_poses = []
        for _ in range(4):
            pallet_pose = self.get_random_pose(env, pallet_size)
            env.add_object(pallet_urdf, pallet_pose, category='fixed')
            pallet_poses.append(pallet_pose)

        # Cube colors.
        colors = [
            utils.COLORS['red'], utils.COLORS['green'], utils.COLORS['blue'], utils.COLORS['yellow']
        ]

        # Add cubes.
        # x, y, z dimensions for the asset size
        cube_size = (0.04, 0.04, 0.04)
        cube_urdf = 'block/block.urdf'

        objs = []
        for i in range(12):
            cube_pose = self.get_random_pose(env, cube_size)
            cube_id = env.add_object(cube_urdf, cube_pose, color=colors[i%4])
            objs.append(cube_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(pallet_pose, i), pallet_pose[1]) for i in place_pos for pallet_pose in pallet_poses]

        # Goal: cubes are sorted by color and stacked in a pyramid in each pallet.
        for i in range(4):
            self.add_goal(objs=objs[i*3:(i+1)*3], matches=np.ones((3, 3)), targ_poses=targs[i*3:(i+1)*3], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / 4, symmetries=[np.pi/2]*3,
                          language_goal=self.lang_template.format(color=list(utils.COLORS.keys())[i]))