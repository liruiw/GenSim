import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class AssemblingKitsSeq(Task):
    """ Precisely place each specified shape in the specified hole following the order prescribed in the
language instruction at each timestep."""

    def __init__(self):
        super().__init__()
        self.max_steps = 7
        self.homogeneous = False

        self.lang_template = "put the {color} {obj} in the {loc}{obj} hole"
        self.task_completed_desc = "done assembling kit."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add kit.
        kit_size = (0.28, 0.2, 0.005)
        kit_urdf = 'kitting/kit.urdf'
        kit_pose = self.get_random_pose(env, kit_size)
        env.add_object(kit_urdf, kit_pose, 'fixed')

        # Shape Names:
        shapes = utils.assembling_kit_shapes
        n_objects = 5
        obj_shapes = self.get_kitting_shapes(n_objects)
        colors, color_names = utils.get_colors(mode=self.mode)

        # Build kit.
        targets = []
        targets_spatial_desc = []
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

            # Decide spatial description based on the location of the hole (top-down view).
            shape_type = obj_shapes[i]
            if list(obj_shapes).count(obj_shapes[i]) > 1:
                duplicate_shapes = [j for j, o in enumerate(obj_shapes) if i != j and o == shape_type]
                other_poses = [utils.apply(kit_pose, targ_pos[d]) for d in duplicate_shapes]

                if all(pos[0] < op[0] and abs(pos[0]-op[0]) > abs(pos[1]-op[1]) for op in other_poses):
                    spatial_desc = "top "
                elif all(pos[0] > op[0] and abs(pos[0]-op[0]) > abs(pos[1]-op[1]) for op in other_poses):
                    spatial_desc = "bottom "
                elif all(pos[1] < op[1] for op in other_poses):
                    spatial_desc = "left "
                elif all(pos[1] > op[1] for op in other_poses):
                    spatial_desc = "right "
                else:
                    spatial_desc = "middle "

                targets_spatial_desc.append(spatial_desc)
            else:
                targets_spatial_desc.append("")

        # Add objects.
        objects, matches = self.make_kitting_objects(env, targets=targets, obj_shapes=obj_shapes, n_objects=n_objects, colors=colors)
        target_idxs = list(range(n_objects))
        np.random.shuffle(target_idxs)
        for i in target_idxs:
            language_goal = (self.lang_template.format(color=color_names[i],
                                                             obj=shapes[obj_shapes[i]],
                                                             loc=targets_spatial_desc[i]))
            self.add_goal(objs=[objects[i]], matches=np.int32([[1]]), targ_poses=[targets[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1 / n_objects, language_goal=language_goal)
            
        self.max_steps = n_objects
