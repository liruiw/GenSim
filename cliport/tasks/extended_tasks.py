from cliport.tasks.align_box_corner import AlignBoxCorner
from cliport.tasks.assembling_kits import AssemblingKits
from cliport.tasks.assembling_kits_seq import AssemblingKitsSeq
from cliport.tasks.block_insertion import BlockInsertion
from cliport.tasks.manipulating_rope import ManipulatingRope
from cliport.tasks.align_rope import AlignRope
from cliport.tasks.packing_boxes import PackingBoxes
from cliport.tasks.packing_shapes import PackingShapes
from cliport.tasks.packing_boxes_pairs import PackingBoxesPairs
from cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from cliport.tasks.palletizing_boxes import PalletizingBoxes
from cliport.tasks.place_red_in_green import PlaceRedInGreen
from cliport.tasks.put_block_in_bowl import PutBlockInBowl
from cliport.tasks.stack_block_pyramid import StackBlockPyramid
from cliport.tasks.stack_block_pyramid_seq import StackBlockPyramidSeq
from cliport.tasks.sweeping_piles import SweepingPiles
from cliport.tasks.separating_piles import SeparatingPiles
from cliport.tasks.task import Task
from cliport.tasks.towers_of_hanoi import TowersOfHanoi
from cliport.tasks.towers_of_hanoi_seq import TowersOfHanoiSeq
from cliport.tasks.generated_task import GeneratedTask

import pybullet as p
import os
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

##################### block insertion
class BlockInsertionTranslation(BlockInsertion):
    """Insertion Task - Translation Variant."""

    def get_random_pose(self, env, obj_size):
        pose = super(BlockInsertionTranslation, self).get_random_pose(env, obj_size)
        pos, rot = pose
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
        return pos, rot

class BlockInsertionEasy(BlockInsertionTranslation):
    """Insertion Task - Easy Variant."""

    def add_block(self, env):
        """Add L-shaped block in fixed position."""
        # size = (0.1, 0.1, 0.04)
        urdf = 'insertion/ell.urdf'
        pose = ((0.5, 0, 0.02), p.getQuaternionFromEuler((0, 0, np.pi / 2)))
        return env.add_object(urdf, pose)

class BlockInsertionSixDof(BlockInsertion):
    """Insertion Task - 6DOF Variant."""

    def __init__(self):
        super().__init__()
        self.sixdof = True
        self.pos_eps = 0.02

    def add_fixture(self, env):
        """Add L-shaped fixture to place block."""
        size = (0.1, 0.1, 0.04)
        urdf = 'insertion/fixture.urdf'
        pose = self.get_random_pose_6dof(env, size)
        env.add_object(urdf, pose, 'fixed')
        return pose

    def get_random_pose_6dof(self, env, obj_size):
        pos, rot = super(BlockInsertionSixDof, self).get_random_pose(env, obj_size)
        z = (np.random.rand() / 10) + 0.03
        pos = (pos[0], pos[1], obj_size[2] / 2 + z)
        roll = (np.random.rand() - 0.5) * np.pi / 2
        pitch = (np.random.rand() - 0.5) * np.pi / 2
        yaw = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
        return pos, rot


class BlockInsertionNoFixture(BlockInsertion):
    """Insertion Task - No Fixture Variant."""

    def add_fixture(self, env):
        """Add target pose to place block."""
        size = (0.1, 0.1, 0.04)
        # urdf = 'insertion/fixture.urdf'
        pose = self.get_random_pose(env, size)
        return pose

# AssemblingKits
class AssemblingKitsSeqUnseenColors(AssemblingKitsSeq):
    """Kitting Task - Easy variant."""
    def __init__(self):
        super().__init__()
        self.mode = 'test'

class AssemblingKitsSeqSeenColors(AssemblingKitsSeqUnseenColors):
    """Kitting Task - Easy variant."""
    def __init__(self):
        super().__init__()
        self.mode = 'train'

class AssemblingKitsSeqFull(AssemblingKitsSeqUnseenColors):
    """Kitting Task - Easy variant."""
    def __init__(self):
        super().__init__()
        self.mode = 'full'


class AssemblingKitsEasy(AssemblingKits):
    """Kitting Task - Easy variant."""

    def __init__(self):
        super().__init__()
        self.rot_eps = np.deg2rad(30)
        self.train_set = np.int32(
            [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19])
        self.test_set = np.int32([3, 11])
        self.homogeneous = True


# PackingBoxesPairs
class PackingBoxesPairsUnseenColors(PackingBoxesPairs):
    def __init__(self):
        super().__init__()
        self.mode = 'test'

class PackingBoxesPairsSeenColors(PackingBoxesPairsUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'train'

class PackingBoxesPairsFull(PackingBoxesPairsUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'all'


# PackingUnseenGoogleObjects
class PackingUnseenGoogleObjectsSeq(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()

    def get_object_names(self):
        return utils.google_seen_obj_shapes

class PackingSeenGoogleObjectsGroup(PackingSeenGoogleObjectsSeq):
    """Packing Seen Google Objects Group task."""

    def __init__(self):
        super().__init__()
        self.lang_template = "pack all the {obj} objects in the brown box"
        self.max_steps = 3

    def choose_objects(self, object_names, k):
        # Randomly choose a category to repeat.
        chosen_objects = np.random.choice(object_names, k, replace=True)
        repeat_category, distractor_category = np.random.choice(chosen_objects, 2, replace=False)
        num_repeats = np.random.randint(2, 3)
        chosen_objects[:num_repeats] = repeat_category
        chosen_objects[num_repeats:2*num_repeats] = distractor_category

        return chosen_objects, repeat_category

    def set_goals(self, object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size):
        # Pack all objects of the chosen (repeat) category.
        num_pack_objs = object_descs.count(repeat_category)
        true_poses = []

        chosen_obj_pts = dict()
        chosen_obj_ids = []
        for obj_idx, (object_id, info) in enumerate(object_ids):
            if object_descs[obj_idx] == repeat_category:
                true_poses.append(zone_pose)
                chosen_obj_pts[object_id] = object_points[object_id]
                chosen_obj_ids.append((object_id, info))

        self.goals.append((
            chosen_obj_ids, np.eye(len(chosen_obj_ids)), true_poses, False, True, 'zone',
            (chosen_obj_pts, [(zone_pose, zone_size)]), 1))
        self.lang_goals.append(self.lang_template.format(obj=repeat_category))

        # Only one mistake allowed.
        self.max_steps = num_pack_objs+1

class PackingUnseenGoogleObjectsGroup(PackingSeenGoogleObjectsGroup):
    """Packing Unseen Google Objects Group task."""

    def __init__(self):
        super().__init__()

    def get_object_names(self):
        return utils.google_unseen_obj_shapes


# PutBlockInBowl
class PutBlockInBowlUnseenColors(PutBlockInBowl):
    def __init__(self):
        super().__init__()
        self.mode = 'test'

class PutBlockInBowlSeenColors(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'train'

class PutBlockInBowlFull(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'full'

# SeparatingPiles
class SeparatingPilesUnseenColors(SeparatingPiles):
    def __init__(self):
        super().__init__()
        self.mode = 'test'

class SeparatingPilesSeenColors(SeparatingPilesUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'train'

class SeparatingPilesFull(SeparatingPilesUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'full'


# StackBlockPyramid
class StackBlockPyramidSeqUnseenColors(StackBlockPyramidSeq):
    def __init__(self):
        super().__init__()
        self.mode = 'test'


class StackBlockPyramidSeqSeenColors(StackBlockPyramidSeqUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'train'

class StackBlockPyramidSeqFull(StackBlockPyramidSeqUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'full'

# TowersOfHanoiSeq

class TowersOfHanoiSeqUnseenColors(TowersOfHanoiSeq):
    def __init__(self):
        super().__init__()
        self.mode = 'test'

class TowersOfHanoiSeqSeenColors(TowersOfHanoiSeqUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'train'

class TowersOfHanoiSeqFull(TowersOfHanoiSeqUnseenColors):
    def __init__(self):
        super().__init__()
        self.mode = 'full'