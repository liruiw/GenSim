import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class TowersOfHanoi(Task):
    """Sequentially move disks from one tower to another—only smaller disks can be on top of larger ones."""

    def __init__(self):
        super().__init__()
        self.max_steps = 14
        self.lang_template = "solve towers of hanoi"
        self.task_completed_desc = "solved towers of hanoi."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add stand.
        base_size = (0.12, 0.36, 0.01)
        base_urdf = 'hanoi/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Rod positions in base coordinates.
        rod_position = ((0, -0.12, 0.03), (0, 0, 0.03), (0, 0.12, 0.03))

        # Add disks.
        disks = []
        n_disks = 3
        for i in range(n_disks):
            disk_urdf = 'hanoi/disk%d.urdf' % i
            pos = utils.apply(base_pose, rod_position[0])
            z = 0.015 * (n_disks - i - 2)
            pos = (pos[0], pos[1], pos[2] + z)
            disks.append(env.add_object(disk_urdf, (pos, base_pose[1])))

        # Solve Hanoi sequence with dynamic programming.
        hanoi_steps = utils.solve_hanoi_all(n_disks)

        # Goal: pick and place disks using Hanoi sequence.
        for step in hanoi_steps:
            disk_id = disks[step[0]]
            targ_position = rod_position[step[2]]
            targ_position = utils.apply(base_pose, targ_position)
            targ_pose = (targ_position, (0, 0, 0, 1))
            self.add_goal(objs=[disk_id], matches=np.int32([[1]]), targ_poses=[targ_pose], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1 / len(hanoi_steps),
                    symmetries=[0], language_goal=self.lang_template)