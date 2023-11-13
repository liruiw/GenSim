"""Base Task class."""

import collections
import os
import random
import string
import tempfile

import cv2
import numpy as np
from cliport.tasks import cameras
from cliport.tasks import primitives
from cliport.tasks.grippers import Suction
from cliport.utils import utils
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
import pybullet as p
from typing import Tuple, List
import re

class Task():
    """Base Task class."""

    def __init__(self):
        self.ee = Suction
        self.mode = 'train'
        self.sixdof = False
        self.primitive = primitives.PickPlace()
        self.oracle_cams = cameras.Oracle.CONFIG
        self.rng = None

        # Evaluation epsilons (for pose evaluation metric).
        self.pos_eps = 0.01
        self.rot_eps = np.deg2rad(15)

        # for piles
        self.num_blocks = 50

        # Workspace bounds.
        self.pix_size = 0.003125
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])
        self.zone_bounds = np.copy(self.bounds)

        self.goals = []
        self.lang_goals = []
        self.obj_points_cache = {}

        self.task_completed_desc = "task completed."
        self.progress = 0
        self._rewards = 0

        self.train_set = np.arange(0, 14)
        self.test_set = np.arange(14, 20)
        self.assets_root = None
        self.homogeneous = False

    def reset(self, env):
        if not self.assets_root:
            raise ValueError('assets_root must be set for task, '
                             'call set_assets_root().')
        self.goals = []
        self.lang_goals = []
        self.progress = 0  # Task progression metric in range [0, 1].
        self._rewards = 0  # Cumulative returned rewards.
        self.obj_points_cache = {}

    def additional_reset(self):
        # Additional changes to make the environment adaptable
        if 'bowl' in self.lang_template:
            # IMPORTANT: increase position tolerance for bowl placement
            self.pos_eps = 0.05

        if 'piles' in self.lang_template:
            # IMPORTANT: Define the primitive to be push and ee to be spatula for tasks involving piles
            self.ee = Spatula
            self.primitive = primitives.push

        if 'rope' in self.lang_template:
            self.primitive = primitives.PickPlace(height=0.02, speed=0.001)
            self.pos_eps = 0.02

    # -------------------------------------------------------------------------
    # Oracle Agent
    # -------------------------------------------------------------------------

    def oracle(self, env):
        """Oracle agent."""
        OracleAgent = collections.namedtuple('OracleAgent', ['act'])

        def act(obs, info):
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            _, hmap, obj_mask = self.get_true_image(env)

            # Unpack next goal step.
            objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]

            for j, targ in enumerate(targs):
                # add default orientation if missing
                if len(targ) == 3 and (type(targs[j][0]) is float or type(targs[j][0]) is np.float32):
                    targs[j] = (targs[j], (0,0,0,1))

            # Match objects to targets without replacement.
            if not replace:

                # Modify a copy of the match matrix.
                matches = matches.copy()

                # Ignore already matched objects.
                for i in range(len(objs)):
                    if type(objs[i]) is int:
                        objs[i] = (objs[i], (False, None))

                    object_id, (symmetry, _) = objs[i]
                    pose = p.getBasePositionAndOrientation(object_id)
                    targets_i = np.argwhere(matches[i, :]).reshape(-1)
                    for j in targets_i:
                        if self.is_match(pose, targs[j], symmetry):
                            matches[i, :] = 0
                            matches[:, j] = 0

            # Get objects to be picked (prioritize farthest from nearest neighbor).
            nn_dists = []
            nn_targets = []
            for i in range(len(objs)):
                if type(objs[i]) is int:
                    objs[i] = (objs[i], (False, None))

                object_id, (symmetry, _) = objs[i]
                xyz, _ = p.getBasePositionAndOrientation(object_id)
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                if len(targets_i) > 0:

                    targets_xyz = np.float32([targs[j][0] for j in targets_i])
                    dists = np.linalg.norm(
                        targets_xyz - np.float32(xyz).reshape(1, 3), axis=1)
                    nn = np.argmin(dists)
                    nn_dists.append(dists[nn])
                    nn_targets.append(targets_i[nn])

                # Handle ignored objects.
                else:
                    nn_dists.append(0)
                    nn_targets.append(-1)
            order = np.argsort(nn_dists)[::-1]

            # Filter out matched objects.
            order = [i for i in order if nn_dists[i] > 0]

            pick_mask = None
            for pick_i in order:
                pick_mask = np.uint8(obj_mask == objs[pick_i][0])

                # Erode to avoid picking on edges.
                pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

                if np.sum(pick_mask) > 0:
                    break

            # Trigger task reset if no object is visible.
            if pick_mask is None or np.sum(pick_mask) == 0:
                self.goals = []
                self.lang_goals = []
                print('Object for pick is not visible. Skipping demonstration.')
                return

            # Get picking pose.
            pick_prob = np.float32(pick_mask)
            pick_pix = utils.sample_distribution(pick_prob)
            # For "deterministic" demonstrations on insertion-easy, use this:
            pick_pos = utils.pix_to_xyz(pick_pix, hmap,
                                        self.bounds, self.pix_size)
            pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

            # Get placing pose.
            targ_pose = targs[nn_targets[pick_i]]
            obj_pose = p.getBasePositionAndOrientation(objs[pick_i][0])
            if not self.sixdof:
                obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
                obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
                obj_pose = (obj_pose[0], obj_quat)
            world_to_pick = utils.invert(pick_pose)
            obj_to_pick = utils.multiply(world_to_pick, obj_pose)
            pick_to_obj = utils.invert(obj_to_pick)

            if len(targ_pose) == 3 and (type(targ_pose[0]) is float or type(targ_pose[0]) is np.float32):
                # add default orientation if missing
                targ_pose = (targ_pose, (0,0,0,1))

            place_pose = utils.multiply(targ_pose, pick_to_obj)

            # Rotate end effector?
            if not rotations:
                place_pose = (place_pose[0], (0, 0, 0, 1))

            place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

            return {'pose0': pick_pose, 'pose1': place_pose}

        return OracleAgent(act)

    # -------------------------------------------------------------------------
    # Reward Function and Task Completion Metrics
    # -------------------------------------------------------------------------

    def reward(self):
        """Get delta rewards for current timestep.

        Returns:
          A tuple consisting of the scalar (delta) reward.
        """
        reward, info = 0, {}

        # Unpack next goal step.
        objs, matches, targs, replace, _, metric, params, max_reward = self.goals[0]

        # Evaluate by matching object poses.
        step_reward = 0

        if metric == 'pose':
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                pose = p.getBasePositionAndOrientation(object_id)
                targets_i = np.argwhere(matches[i, :])
                if len(targets_i) > 0:
                    targets_i = targets_i.reshape(-1)
                    for j in targets_i:
                        target_pose = targs[j]
                        if self.is_match(pose, target_pose, symmetry):
                            step_reward += max_reward / len(objs)
                            print(f"object {i} match with target {j} rew: {step_reward:.3f}")
                            break

        # Evaluate by measuring object intersection with zone.
        elif metric == 'zone':
            zone_pts, total_pts = 0, 0
            zones = params

            if len(self.obj_points_cache) == 0 or objs[0][0] not in self.obj_points_cache:
                for obj_id, _ in objs:
                    self.obj_points_cache[obj_id] = self.get_box_object_points(obj_id)

            for zone_idx, (zone_pose, zone_size) in enumerate(zones):
                # Count valid points in zone.
                for (obj_id, _) in objs:
                    pts = self.obj_points_cache[obj_id]
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    world_to_zone = utils.invert(zone_pose)
                    obj_to_zone = utils.multiply(world_to_zone, obj_pose)
                    pts = np.float32(utils.apply(obj_to_zone, pts))

                    if len(zone_size) > 1:
                        valid_pts = np.logical_and.reduce([
                            pts[0, :] > -zone_size[0] / 2, pts[0, :] < zone_size[0] / 2,
                            pts[1, :] > -zone_size[1] / 2, pts[1, :] < zone_size[1] / 2,
                            pts[2, :] < self.zone_bounds[2, 1]])

                    zone_pts += np.sum(np.float32(valid_pts))
                    total_pts += pts.shape[1]

            if total_pts > 0:
                step_reward = max_reward * (zone_pts / total_pts)

        # Get cumulative rewards and return delta.
        reward = self.progress + step_reward - self._rewards
        self._rewards = self.progress + step_reward

        # Move to next goal step if current goal step is complete.
        if np.abs(max_reward - step_reward) < 0.01:
            self.progress += max_reward  # Update task progress.
            self.goals.pop(0)
            if len(self.lang_goals) > 0:
                self.lang_goals.pop(0)

        return reward, info

    def done(self):
        """Check if the task is done or has failed.

        Returns:
          True if the episode should be considered a success.
        """
        return (len(self.goals) == 0) or (self._rewards > 0.99)
        # return zone_done or defs_done or goal_done

    # -------------------------------------------------------------------------
    # Environment Helper Functions
    # -------------------------------------------------------------------------

    def is_match(self, pose0, pose1, symmetry):
        """Check if pose0 and pose1 match within a threshold.
        pose0 and pose1 should both be tuples of (translation, rotation).
        Return true if the pose translation and orientation errors are below certain thresholds"""
        if len(pose1) == 3 and (not hasattr(pose1[0], '__len__')):
            # add default orientation if missing
            pose1 = (pose1, (0,0,0,1))
        # print(len(pose1) == 3, not hasattr(pose1[0], '__len__'))
        # print(pose1, pose0)
        # Get translational error.
        diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
        dist_pos = np.linalg.norm(diff_pos)

        # Get rotational error around z-axis (account for symmetries).
        diff_rot = 0
        if symmetry > 0:
            rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
            rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
            diff_rot = np.abs(rot0 - rot1) % symmetry
            if diff_rot > (symmetry / 2):
                diff_rot = symmetry - diff_rot

        return (dist_pos < self.pos_eps) and (diff_rot < self.rot_eps)

    def get_true_image(self, env):
        """Get RGB-D orthographic heightmaps and segmentation masks."""

        # Capture near-orthographic RGB-D images and segmentation masks.
        color, depth, segm = env.render_camera(self.oracle_cams[0])

        # Combine color with masks for faster processing.
        color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

        # Reconstruct real orthographic projection from point clouds.
        hmaps, cmaps = utils.reconstruct_heightmaps(
            [color], [depth], self.oracle_cams, self.bounds, self.pix_size)

        # Split color back into color and masks.
        cmap = np.uint8(cmaps)[0, Ellipsis, :3]
        hmap = np.float32(hmaps)[0, Ellipsis]
        mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
        return cmap, hmap, mask

    def get_random_pose(self, env, obj_size=0.1, **kwargs) -> (List, List):
        """
        Get random collision-free object pose within workspace bounds.
        :param obj_size: (3, ) contains the object size in x,y,z dimensions
        return: translation (3, ), rotation (4, ) """

        # Get erosion size of object in pixels.
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self.pix_size))

        _, hmap, obj_mask = self.get_true_image(env)

        # Randomly sample an object pose within free-space pixels.
        free = np.ones(obj_mask.shape, dtype=np.uint8)
        for obj_ids in env.obj_ids.values():
            for obj_id in obj_ids:
                free[obj_mask == obj_id] = 0
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))

        # if np.sum(free) == 0:
        #     return None, None

        if np.sum(free) == 0:
            # avoid returning None
            pix = (obj_mask.shape[0] // 2, obj_mask.shape[1] // 2)
        else:
            pix = utils.sample_distribution(np.float32(free))
        pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)

        if len(obj_size) == 2:
            print("Should have z dimension in obj_size as well.")
            pos = [pos[0], pos[1], 0.05]
        else:
            pos = [pos[0], pos[1], obj_size[2] / 2]
        theta = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot

    def get_lang_goal(self):
        if len(self.lang_goals) == 0:
            return self.task_completed_desc
        else:
            return self.lang_goals[0]

    def get_reward(self):
        return float(self._rewards)
    
    def add_corner_anchor_for_pose(self, env, pose):
        corner_template = 'corner/corner-template.urdf'
        replace = {'DIMX': (0.04,), 'DIMY': (0.04,)}

        # IMPORTANT: REPLACE THE TEMPLATE URDF
        corner_urdf = self.fill_template(corner_template, replace)
        if len(pose) != 2:
            pose = [pose,(0,0,0,1)]
        env.add_object(corner_urdf, pose, 'fixed')


    def get_target_sample_surface_points(self, model, scale, pose, num_points=50):
        import trimesh
        mesh = trimesh.load_mesh(model)
        points = trimesh.sample.volume_mesh(mesh, num_points * 3)
        points = points[:num_points]
        points = points * np.array(scale)
        points = utils.apply(pose, points.T)
        poses = [((x,y,z),(0,0,0,1)) for x, y, z in zip(points[0], points[1], points[2])]
        return poses
    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def check_require_obj(self, path):
        return os.path.exists(path.replace(".urdf", ".obj"))

    def fill_template(self, template, replace):
        """Read a file and replace key strings.
        NOTE: This function must be called if a URDF has template in its name """

        full_template_path = os.path.join(self.assets_root, template)
        if not os.path.exists(full_template_path) or (self.check_require_obj(full_template_path) and 'template' not in full_template_path):
            return template

        with open(full_template_path, 'r') as file:
            fdata = file.read()

        for field in replace:
            # if  not hasattr(replace[field], '__len__'):
            #     replace[field] = (replace[field], )

            for i in range(len(replace[field])):
                fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))

            if field == 'COLOR':
                # handle gpt
                pattern = r'<color rgba="(.*?)"/>'
                code_string = re.findall(pattern, fdata)
                if type(replace[field]) is str:
                    replace[field] = utils.COLORS[replace[field]]
                for to_replace_color in  code_string:
                    fdata = fdata.replace(f'{to_replace_color}', " ".join([str(x) for x in list(replace[field]) + [1]]))
            
        alphabet = string.ascii_lowercase + string.digits
        rname = ''.join(random.choices(alphabet, k=16))
        tmpdir = tempfile.gettempdir()
        template_filename = os.path.split(template)[-1]
        fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
        with open(fname, 'w') as file:
            file.write(fdata)
        return fname

    def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z) -> Tuple:
        """Get random box size."""
        size = np.random.rand(3)
        size[0] = size[0] * (max_x - min_x) + min_x
        size[1] = size[1] * (max_y - min_y) + min_y
        size[2] = size[2] * (max_z - min_z) + min_z
        return tuple(size)

    def get_box_object_points(self, obj):
        obj_shape = p.getVisualShapeData(obj)
        obj_dim = obj_shape[0][3]
        obj_dim = tuple(d for d in obj_dim)
        xv, yv, zv = np.meshgrid(
            np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
            np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
            np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
            sparse=False, indexing='xy')
        return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

    def get_sphere_object_points(self, obj):
        return self.get_box_object_points(obj)

    def get_mesh_object_points(self, obj):
        mesh = p.getMeshData(obj)
        mesh_points = np.array(mesh[1])
        mesh_dim = np.vstack((mesh_points.min(axis=0), mesh_points.max(axis=0)))
        xv, yv, zv = np.meshgrid(
            np.arange(mesh_dim[0][0], mesh_dim[1][0], 0.02),
            np.arange(mesh_dim[0][1], mesh_dim[1][1], 0.02),
            np.arange(mesh_dim[0][2], mesh_dim[1][2], 0.02),
            sparse=False, indexing='xy')
        return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

    def color_random_brown(self, obj):
        shade = np.random.rand() + 0.5
        color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
        p.changeVisualShape(obj, -1, rgbaColor=color)

    def set_assets_root(self, assets_root):
        self.assets_root = assets_root

    def zip_obj_ids(self, obj_ids, symmetries):
        if type(obj_ids[0]) is tuple:
            return obj_ids

        if  symmetries is None:
             symmetries = [0.] * len(obj_ids)
        objs = []

        for obj_id, symmetry in zip(obj_ids, symmetries):
            objs.append((obj_id, (symmetry, None)))
        return objs

    def add_goal(self, objs, matches, targ_poses, replace, rotations, metric, params, step_max_reward,
                        symmetries=None, language_goal=None, **kwargs):
        """ Add the goal to the environment
        - objs (List of Tuple [(obj_id, (float, None))] ): object ID, (the radians that the object is symmetric over, None). Do not pass in `(object id, object pose)` as the wrong tuple. or `object id` (such as `containers[i][0]`).
        - matches (Binary Matrix): a binary matrix that denotes which object is matched with which target. This matrix has dimension len(objs) x len(targ_poses).
        - targ_poses (List of Poses [(translation, rotation)] ): a list of target poses of tuple (translation, rotation). Don't pass in object IDs such as `bowls[i-1][0]` or  `[stands[i][0]]`.
        - replace (Boolean): whether each object can match with one unique target.   This is important if we have one target and multiple objects. If it's set to be false, then any object matching with the target will satisfy.
        - rotations (Boolean): whether the placement action has a rotation degree of freedom.
        - metric (`pose` or `zone`): `pose` or `zone` that the object needs to be transported to. Example: `pose`.
        - params ([(zone_target, zone_size)])): has to be [(zone_target, zone_size)] if the metric is `zone` where obj_pts is a dictionary that maps object ID to points.
        - step_max_reward (float): the maximum reward of matching all the objects with all the target poses.
        """
        objs = self.zip_obj_ids(objs, symmetries)
        self.goals.append((objs, matches, targ_poses, replace, rotations,
                           metric, params, step_max_reward))
        if language_goal is not None:
            if type(language_goal) is str:
                self.lang_goals.append(language_goal)
            elif type(language_goal) is list:
                self.lang_goals.extend(language_goal)

    def make_piles(self, env, block_color=None, *args, **kwargs):
        """
        add the piles objects for tasks involving piles
        """
        obj_ids = []
        for _ in range(self.num_blocks):
            rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
            ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
            xyz = (rx, ry, 0.01)
            theta = np.random.rand() * 2 * np.pi
            xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            obj_id = env.add_object('block/small.urdf', (xyz, xyzw))
            if block_color is not None:
                if type(block_color) is str:
                    block_color = utils.COLORS[block_color]
                p.changeVisualShape(obj_id, -1, rgbaColor=block_color + [1])

            obj_ids.append(obj_id)
        return obj_ids

    def make_rope(self, *args, **kwargs):
        return self.make_ropes(*args, **kwargs)

    def make_ropes(self, env, corners, radius=0.005, n_parts=20, color_name='red', *args, **kwargs):
        """ add cables simulation for tasks that involve cables """
        # Get corner points of square.
        
        # radius = 0.005
        length = 2 * radius * n_parts * np.sqrt(2)
        corner0, corner1 = corners
        # Add cable (series of articulated small blocks).
        increment = (np.float32(corner1) - np.float32(corner0)) / n_parts
        position, _ = self.get_random_pose(env, (0.1, 0.1, 0.1))
        position = np.float32(position)
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)
        parent_id = -1
        targets = []
        objects = []

        for i in range(n_parts):
            position[2] += np.linalg.norm(increment)
            part_id = p.createMultiBody(0.1, part_shape, part_visual,
                                        basePosition=position)
            if parent_id > -1:
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=parent_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=part_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, np.linalg.norm(increment)),
                    childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)

            if (i > 0) and (i < n_parts - 1):
                color = utils.COLORS[color_name] + [1]
                p.changeVisualShape(part_id, -1, rgbaColor=color)

            env.obj_ids['rigid'].append(part_id)
            parent_id = part_id
            target_xyz = np.float32(corner0) + i * increment + increment / 2
            objects.append((part_id, (0, None)))
            targets.append((target_xyz, (0, 0, 0, 1)))

            if  hasattr(env, 'record_cfg') and 'blender_render' in env.record_cfg and env.record_cfg['blender_render']:
                sphere_template = os.path.join(self.assets_root, 'sphere/sphere_rope.urdf')
                env.blender_recorder.register_object(part_id, os.path.join(self.assets_root, 'sphere/sphere_rope.urdf'))


        matches = np.clip(np.eye(n_parts) + np.eye(n_parts)[::-1], 0, 1)
        return objects, targets, matches


    def get_kitting_shapes(self, n_objects):
        if self.mode == 'train':
            obj_shapes = np.random.choice(self.train_set, n_objects)
        else:
            if self.homogeneous:
                obj_shapes = [np.random.choice(self.test_set)] * n_objects
            else:
                obj_shapes = np.random.choice(self.test_set, n_objects)

        return obj_shapes


    def make_kitting_objects(self, env, targets, obj_shapes, n_objects, colors):
        symmetry = [
            2 * np.pi, 2 * np.pi, 2 * np.pi / 3, np.pi / 2, np.pi / 2, 2 * np.pi,
            np.pi, 2 * np.pi / 5, np.pi, np.pi / 2, 2 * np.pi / 5, 0, 2 * np.pi,
            2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 2 * np.pi / 6, 2 * np.pi, 2 * np.pi
        ]
        objects = []
        matches = []
        template = 'kitting/object-template.urdf'

        for i in range(n_objects):
            shape = obj_shapes[i]
            size = (0.08, 0.08, 0.02)
            pose = self.get_random_pose(env, size)
            fname = f'{shape:02d}.obj'
            fname = os.path.join(self.assets_root, 'kitting', fname)
            scale = [0.003, 0.003, 0.001]  # .0005
            replace = {'FNAME': (fname,), 'SCALE': scale, 'COLOR': colors[i]}

            # IMPORTANT: REPLACE THE TEMPLATE URDF
            urdf = self.fill_template(template, replace)
            block_id = env.add_object(urdf, pose)
            objects.append((block_id, (symmetry[shape], None)))
            match = np.zeros(len(targets))
            match[np.argwhere(obj_shapes == shape).reshape(-1)] = 1
            matches.append(match)
        return objects, matches

    def spawn_box(self):
        """Palletizing: spawn another box in the workspace if it is empty."""
        workspace_empty = True
        if self.goals:
            for obj in self.goals[0][0]:
                obj_pose = p.getBasePositionAndOrientation(obj[0])
                workspace_empty = workspace_empty and ((obj_pose[0][1] < -0.5) or
                                                       (obj_pose[0][1] > 0))
            if not self.steps:
                self.goals = []
                print('Palletized boxes toppled. Terminating episode.')
                return

            if workspace_empty:
                obj = self.steps[0]
                theta = np.random.random() * 2 * np.pi
                rotation = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
                p.resetBasePositionAndOrientation(obj, [0.5, -0.25, 0.1], rotation)
                self.steps.pop(0)

        # Wait until spawned box settles.
        for _ in range(480):
            p.stepSimulation()

    def get_asset_full_path(self, path):
        return path