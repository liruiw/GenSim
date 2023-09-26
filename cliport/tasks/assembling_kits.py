import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class AssemblingKits(Task):
    """pick up different objects and arrange them on a board marked with corresponding silhouettes."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.homogeneous = False

        self.lang_template = "put all the blocks inside the holes they fit in"
        self.task_completed_desc = "done assembling blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add kit.
        kit_size = (0.28, 0.2, 0.005)
        kit_urdf = 'kitting/kit.urdf'
        kit_pose = self.get_random_pose(env, kit_size)
        env.add_object(kit_urdf, kit_pose, 'fixed')

        n_objects = 5
        obj_shapes = self.get_kitting_shapes(n_objects)
        colors = [
            utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['red']
        ]

        # Build kit.
        targets = []
        targ_pos = [[-0.09, 0.045, 0.0014], [0, 0.045, 0.0014],
                    [0.09, 0.045, 0.0014], [-0.045, -0.045, 0.0014],
                    [0.045, -0.045, 0.0014]]
        template = 'kitting/object-template.urdf'

        for i in range(n_objects):
            shape = os.path.join(self.assets_root, 'kitting',
                                 f'{obj_shapes[i]:02d}.obj')
            scale = [0.003, 0.003, 0.0001]  # .0005
            pos = utils.apply(kit_pose, targ_pos[i])
            theta = np.random.rand() * 2 * np.pi
            rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': [0.2, 0.2, 0.2]}

            # IMPORTANT: REPLACE THE TEMPLATE URDF
            urdf = self.fill_template(template, replace)
            env.add_object(urdf, (pos, rot), 'fixed')
            targets.append((pos, rot))

        # Add objects.
        objects, matches = self.make_kitting_objects(env, targets=targets, obj_shapes=obj_shapes, n_objects=n_objects, colors=colors)
        matches = np.int32(matches)
        self.add_goal(objs=objects, matches=matches, targ_poses=targets, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1, language_goal=self.lang_template)
 