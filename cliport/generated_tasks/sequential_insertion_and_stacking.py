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

class SequentialInsertionAndStacking(Task):
    """Pick up and insert each ell block into the corresponding colored fixture in the sequence of red, blue, and green. After successful insertion, pick up the three blocks again from the fixtures and stack them in a corner of the tabletop in the same color sequence - red at the bottom, blue in the middle, and green on top."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "insert the {color} ell block into the {color} fixture and then stack them in the corner"
        self.task_completed_desc = "done inserting and stacking."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add fixtures.
        fixture_size = (0.12, 0.12, 0)
        fixture_urdf = 'insertion/fixture.urdf'
        fixture_poses = []
        colors = ['red', 'blue', 'green']
        for color in colors:
            fixture_pose = self.get_random_pose(env, fixture_size)
            env.add_object(fixture_urdf, fixture_pose, category='fixed', color=utils.COLORS[color])
            fixture_poses.append(fixture_pose)

        # Add ell blocks.
        ell_size = (0.04, 0.04, 0.04)
        ell_urdf = 'insertion/ell.urdf'
        ells = []
        for color in colors:
            ell_pose = self.get_random_pose(env, ell_size)
            ell_id = env.add_object(ell_urdf, ell_pose, color=utils.COLORS[color])
            ells.append(ell_id)

        # Goal: each ell block is in the corresponding colored fixture.
        for i in range(3):
            self.add_goal(objs=[ells[i]], matches=np.ones((1, 1)), targ_poses=[fixture_poses[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/3)
            self.lang_goals.append(self.lang_template.format(color=colors[i]))

        # Add corner.
        corner_size = (0.12, 0.12, 0)
        corner_pose = self.get_random_pose(env, corner_size)
        corner_urdf = 'corner/corner-template.urdf'
        env.add_object(corner_urdf, corner_pose, category='fixed')

        # Goal: ell blocks are stacked in the corner in the color sequence - red at the bottom, blue in the middle, and green on top.
        stack_poses = [(0, 0, 0.04), (0, 0, 0.08), (0, 0, 0.12)]
        targs = [(utils.apply(corner_pose, i), corner_pose[1]) for i in stack_poses]
        self.add_goal(objs=ells, matches=np.ones((3, 3)), targ_poses=targs, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/3,
                          language_goal="stack the ell blocks in the corner in the color sequence - red at the bottom, blue in the middle, and green on top")