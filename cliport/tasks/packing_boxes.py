import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import pybullet as p


class PackingBoxes(Task):
    """pick up randomly sized boxes and place them tightly into a container."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "pack all the boxes inside the brown box"
        self.task_completed_desc = "done packing boxes."

        self.zone_bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.08]])
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add container box.
        zone_size = self.get_random_size(0.05, 0.3, 0.05, 0.3, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        replace = {'DIM': zone_size, 'HALF': (zone_size[0] / 2, zone_size[1] / 2, zone_size[2] / 2)}

        # IMPORTANT: REPLACE THE TEMPLATE URDF
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')

        margin = 0.01
        min_object_dim = 0.05
        bboxes = []

        # Split container space with KD trees.
        stack_size = np.array(zone_size)
        stack_size[0] -= 0.01
        stack_size[1] -= 0.01
        root_size = (0.01, 0.01, 0) + tuple(stack_size)
        root = utils.TreeNode(None, [], bbox=np.array(root_size))
        utils.KDTree(root, min_object_dim, margin, bboxes)

        colors = [utils.COLORS[c] for c in utils.COLORS if c != 'brown']

        # Add objects in container.
        object_ids = []
        bboxes = np.array(bboxes)
        object_template = 'box/box-template.urdf'

        # Compute object points that are needed for zone
        for bbox in bboxes:
            size = bbox[3:] - bbox[:3]
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2
            pose = (position, (0, 0, 0, 1))
            pose = utils.multiply(zone_pose, pose)

            # IMPORTANT: REPLACE THE TEMPLATE URDF
            urdf = self.fill_template(object_template, {'DIM': size})
            icolor = np.random.choice(range(len(colors)), 1).squeeze()
            box_id = env.add_object(urdf, pose, color=colors[icolor])
            object_ids.append(box_id)

        # Randomly select object in box and save ground truth pose.
        object_volumes = []
        true_poses = []

        for object_id in object_ids:
            true_pose = p.getBasePositionAndOrientation(object_id)
            object_size = p.getVisualShapeData(object_id)[0][3]
            object_volumes.append(np.prod(np.array(object_size) * 100))
            pose = self.get_random_pose(env, object_size)
            p.resetBasePositionAndOrientation(object_id, pose[0], pose[1])
            true_poses.append(true_pose)

        self.add_goal(objs=object_ids, matches=np.eye(len(object_ids)), targ_poses=true_poses, replace=False,
                rotations=True, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1, language_goal=self.lang_template)