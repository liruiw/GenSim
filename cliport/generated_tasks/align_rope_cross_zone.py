import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula

class AlignRopeCrossZone(Task):
    """Align a deformable rope across the diagonal of a zone marked on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "align the rope across the diagonal of a zone"
        self.task_completed_desc = "done aligning."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zone.
        length = 0.12
        zone_size = (length, length, 0.01)
        zone_pose = self.get_random_pose(env, zone_size)
        zone_urdf = 'zone/zone.urdf'
        env.add_object(zone_urdf, zone_pose, 'fixed')

        # Add rope.
        rope_size  = (length, 0.01, 0.01)
        rope_pose = self.get_random_pose(env, rope_size)
        corner1_pose = utils.apply(zone_pose, (length / 2, length / 2, 0.01))
        corner2_pose = utils.apply(zone_pose, (-length / 2, -length / 2, 0.01))
        rope_id, targets, matches = self.make_rope(env, (corner1_pose, corner2_pose), n_parts=10)

        # Goal: rope is aligned with the diagonal of the zone.
        self.add_goal(objs=rope_id, matches=matches, targ_poses=targets, replace=False,
                rotations=False, metric='pose', params=None, step_max_reward=1, language_goal=self.lang_template)
