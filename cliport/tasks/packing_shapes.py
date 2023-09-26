import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class PackingShapes(Task):
    """pick up randomly sized shapes and place them tightly into a container."""

    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.homogeneous = False

        self.lang_template = "pack the {obj} in the brown box"
        self.task_completed_desc = "done packing shapes."
        self.additional_reset()


    def reset(self, env):
        super().reset(env)

        # Shape Names:
        shapes = utils.assembling_kit_shapes

        n_objects = 5
        if self.mode == 'train':
            obj_shapes = np.random.choice(self.train_set, n_objects, replace=False)
        else:
            if self.homogeneous:
                obj_shapes = [np.random.choice(self.test_set, replace=False)] * n_objects
            else:
                obj_shapes = np.random.choice(self.test_set, n_objects, replace=False)

        # Shuffle colors to avoid always picking an object of the same color
        colors, color_names = utils.get_colors(mode=self.mode)

        # Add container box.
        zone_size = self.get_random_size(0.1, 0.15, 0.1, 0.15, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        replace = {'DIM': zone_size, 'HALF': (zone_size[0] / 2, zone_size[1] / 2, zone_size[2] / 2)}
        # IMPORTANT: REPLACE THE TEMPLATE URDF with `fill_template`
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')

        # Add objects.
        objects = []
        template = 'kitting/object-template.urdf'
        for i in range(n_objects):
            shape = obj_shapes[i]

            # x, y, z dimensions for the asset size
            size = (0.08, 0.08, 0.02)
            pose= self.get_random_pose(env, size)
            fname = f'{shape:02d}.obj'
            fname = os.path.join(self.assets_root, 'kitting', fname)
            scale = [0.003, 0.003, 0.001]  # .0005
            replace = {'FNAME': (fname,),
                       'SCALE': scale,
                       'COLOR': colors[i]}

            # IMPORTANT: REPLACE THE TEMPLATE URDF
            urdf = self.fill_template(template, replace)
            block_id = env.add_object(urdf, pose)
            objects.append(block_id)

        # Pick the first shape.
        num_objects_to_pick = 1
        for i in range(num_objects_to_pick):
            # IMPORTANT: Specify (obj_pts, [(zone_pose, zone_size)]) for target `zone`. obj_pts is a dict
            language_goal = self.lang_template.format(obj=shapes[obj_shapes[i]])
            self.add_goal(objs=[objects[i]], matches=np.int32([[1]]), targ_poses=[zone_pose], replace=False,
                rotations=True, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1 / num_objects_to_pick,
                language_goal=language_goal)