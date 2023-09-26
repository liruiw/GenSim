import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p


class BlockInsertion(Task):
    """pick up the L-shaped red block and place it into the L-shaped fixture."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "put the L shape block in the L shape hole"
        self.task_completed_desc = "done with insertion."
        self.additional_reset()

    def get_random_pose(self, env, obj_size):
        pose = super().get_random_pose(env, obj_size)
        pos, rot = pose
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
        return pos, rot

    def reset(self, env):
        super().reset(env)

        """Add L-shaped block."""
        size = (0.1, 0.1, 0.04)
        urdf = 'insertion/ell.urdf'
        pose = self.get_random_pose(env, size)
        block_id = env.add_object(urdf, pose)

        """Add L-shaped fixture to place block."""
        size = (0.1, 0.1, 0.04)
        urdf = 'insertion/fixture.urdf'
        targ_pose = self.get_random_pose(env, size)
        env.add_object(urdf, targ_pose, 'fixed')

        self.add_goal(objs=[block_id], matches=np.int32([[1]]), targ_poses=[targ_pose], replace=False,
                rotations=False, metric='pose', params=None, step_max_reward=1, symmetries=[2 * np.pi],
                language_goal=self.lang_template)
