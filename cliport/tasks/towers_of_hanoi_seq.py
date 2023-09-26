import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import pybullet as p
import random

class TowersOfHanoiSeq(Task):
    """Move the ring to the specified peg in the language instruction at each timestep"""

    def __init__(self):
        super().__init__()
        self.max_steps = 14
        self.lang_template = "move the {obj} ring to the {loc}"
        self.task_completed_desc = "solved towers of hanoi."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add stand.
        base_size = (0.12, 0.36, 0.01)
        base_urdf = 'hanoi/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Choose three colors for three rings.
        colors, color_names  = utils.get_colors(mode=self.mode, n_colors=3)

        # Rod positions in base coordinates.
        rod_pos = ((0, -0.12, 0.03), (0, 0, 0.03), (0, 0.12, 0.03))
        rod_names = ('lighter brown side', 'middle of the stand', 'darker brown side')

        # Add disks.
        disks = []
        disks_names = {}
        n_disks = 3
        for i in range(n_disks):
            disk_urdf = 'hanoi/disk%d.urdf' % i
            pos = utils.apply(base_pose, rod_pos[0])
            z = 0.015 * (n_disks - i - 2)
            pos = (pos[0], pos[1], pos[2] + z)
            ring_id = env.add_object(disk_urdf, (pos, base_pose[1]), color=colors[i])
            disks.append(ring_id)
            disks_names[ring_id] = color_names[i]

        # Solve Hanoi sequence with dynamic programming.
        hanoi_steps = utils.solve_hanoi_all(n_disks)

        # Goal: pick and place disks using Hanoi sequence.
        for step in hanoi_steps:
            disk_id = disks[step[0]]
            targ_pos = rod_pos[step[2]]
            targ_pos = utils.apply(base_pose, targ_pos)
            targ_pose = (targ_pos, (0, 0, 0, 1))
            language_goal = self.lang_template.format(obj=disks_names[disk_id],
                                                             loc=rod_names[step[2]])
            self.add_goal(objs=[disk_id], matches=np.int32([[1]]), targ_poses=[targ_pose], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / len(hanoi_steps),
                    symmetries=[0] , language_goal=language_goal)