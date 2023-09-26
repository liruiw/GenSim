import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import pybullet as p


class PackingSeenGoogleObjectsSeq(Task):
    """: Place the specified objects in the brown box following the order prescribed in the language
instruction at each timestep."""

    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.lang_template = "pack the {obj} in the brown box"
        self.task_completed_desc = "done packing objects."
        self.object_names = self.get_object_names()
        self.additional_reset()

    def get_object_names(self):
        return utils.google_all_shapes

    def reset(self, env):
        super().reset(env)

        # object names
        object_names = self.object_names[self.mode]

        # Add container box.
        zone_size = self.get_random_size(0.2, 0.35, 0.2, 0.35, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template_DIM_HALF.urdf'
        replace = {'DIM': zone_size, 'HALF': (zone_size[0] / 2, zone_size[1] / 2, zone_size[2] / 2)}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')

        margin = 0.01
        min_object_dim = 0.08
        bboxes = []

        # Split container space with KD trees.
        stack_size = np.array(zone_size)
        stack_size[0] -= 0.01
        stack_size[1] -= 0.01
        root_size = (0.01, 0.01, 0) + tuple(stack_size)
        root = utils.TreeNode(None, [], bbox=np.array(root_size))
        utils.KDTree(root, min_object_dim, margin, bboxes)

        # Add Google Scanned Objects to scene.
        object_ids = []
        bboxes = np.array(bboxes)
        scale_factor = 5
        object_template = 'google/object-template_FNAME_COLOR_SCALE.urdf'
        chosen_objs, repeat_category = self.choose_objects(object_names, len(bboxes))
        object_descs = []
        for i, bbox in enumerate(bboxes):
            size = bbox[3:] - bbox[:3]
            max_size = size.max()
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2
            shape_size = max_size * scale_factor
            pose = self.get_random_pose(env, size)

            # Add object only if valid pose found.
            if pose[0] is not None:
                # Initialize with a slightly tilted pose so that the objects aren't always erect.
                slight_tilt = utils.q_mult(pose[1], (-0.1736482, 0, 0, 0.9848078))
                ps = ((pose[0][0], pose[0][1], pose[0][2]+0.05), slight_tilt)

                object_name = chosen_objs[i]
                object_name_with_underscore = object_name.replace(" ", "_")
                mesh_file = os.path.join(self.assets_root,
                                         'google',
                                         'meshes_fixed',
                                         f'{object_name_with_underscore}.obj')
                texture_file = os.path.join(self.assets_root,
                                            'google',
                                            'textures',
                                            f'{object_name_with_underscore}.png')

                try:
                    replace = {'FNAME': (mesh_file,),
                               'SCALE': [shape_size, shape_size, shape_size],
                               'COLOR': (0.2, 0.2, 0.2)}
                    urdf = self.fill_template(object_template, replace)
                    box_id = env.add_object(urdf, ps)
                    object_ids.append((box_id, (0, None)))

                    texture_id = p.loadTexture(texture_file)
                    p.changeVisualShape(box_id, -1, textureUniqueId=texture_id)
                    p.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1])

                    object_descs.append(object_name)

                except Exception as e:
                    print("Failed to load Google Scanned Object in PyBullet")
                    print(object_name_with_underscore, mesh_file, texture_file)
                    print(f"Exception: {e}")

        self.set_goals(object_descs, object_ids, repeat_category, zone_pose, zone_size)

        for i in range(480):
            p.stepSimulation()

    def choose_objects(self, object_names, k):
        repeat_category = None
        return np.random.choice(object_names, k, replace=False), repeat_category

    def set_goals(self, object_descs, object_ids,  repeat_category, zone_pose, zone_size):
        # Random picking sequence.
        num_pack_objs = np.random.randint(1, len(object_ids))

        object_ids = object_ids[:num_pack_objs]
        true_poses = []
        for obj_idx, (object_id, _) in enumerate(object_ids):
            true_poses.append(zone_pose)
            language_goal = self.lang_template.format(obj=object_descs[obj_idx])
            self.add_goal(objs=[object_id], matches=np.int32([[1]]), targ_poses=[zone_pose], replace=False,
                rotations=True, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1 / len(object_ids),
                language_goal=language_goal)

        # Only mistake allowed.
        self.max_steps = len(object_ids)+1

