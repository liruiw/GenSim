import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula

class AlignRopeAlongLine(Task):
    """Align a deformable rope along a straight line marked on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "align the rope along the line"
        self.task_completed_desc = "done aligning."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add line.
        length = np.random.uniform(0.18, 0.25)
        line_size = (length, 0.01, 0.01)
        line_pose = self.get_random_pose(env, line_size)
        line_template = 'line/line-template.urdf'
        replace = {'DIM': line_size, 'HALF': (line_size[0] / 2, line_size[1] / 2, line_size[2] / 2)}
        line_urdf = self.fill_template(line_template, replace)
        env.add_object(line_urdf, line_pose, 'fixed')

        # Add rope.
        rope_size  = (length, 0.01, 0.01)
        rope_pose = self.get_random_pose(env, rope_size)
        corner1_pose = utils.apply(line_pose, (length / 2, 0.01, 0.01))
        corner2_pose = utils.apply(line_pose, (-length / 2, 0.01, 0.01))
        rope_id, targets, matches = self.make_rope(env, (corner1_pose, corner2_pose), n_parts=15)

        # Goal: rope is aligned with the line.
        self.add_goal(objs=rope_id, matches=matches, targ_poses=targets, replace=False,
                rotations=False, metric='pose', params=None, step_max_reward=1, language_goal=self.lang_template)
