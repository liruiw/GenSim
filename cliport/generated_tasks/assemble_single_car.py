import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class AssembleSingleCar(Task):
    """Assemble a mini car using a large blue box as the body, a smaller red box on top as the roof, and two tiny green boxes on the sides as wheels."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "build a mini car using a large blue box as the body, a smaller red box on top as the roof, and two tiny green boxes on the sides as wheels"
        self.task_completed_desc = "done assembling the car."

    def reset(self, env):
        super().reset(env)

        # Add car body (large blue box).
        body_size = (0.1, 0.05, 0.02)  # x, y, z dimensions
        body_pose = self.get_random_pose(env, body_size)
        body_urdf = 'box/box-template.urdf'
        body_color = utils.COLORS['blue']
        body_id = env.add_object(body_urdf, body_pose, color=body_color)

        # Add car roof (smaller red box).
        roof_size = (0.08, 0.04, 0.02)  # x, y, z dimensions
        roof_pose = self.get_random_pose(env, roof_size)
        roof_urdf = 'box/box-template.urdf'
        roof_color = utils.COLORS['red']
        roof_id = env.add_object(roof_urdf, roof_pose, color=roof_color)

        # Add car wheels (two tiny green boxes).
        wheel_size = (0.02, 0.02, 0.01)  # x, y, z dimensions
        wheel_urdf = 'box/box-template.urdf'
        wheel_color = utils.COLORS['green']
        wheel_ids = []
        
        for _ in range(2):
            wheel_pose = self.get_random_pose(env, wheel_size)
            wheel_id = env.add_object(wheel_urdf, wheel_pose, color=wheel_color)
            wheel_ids.append(wheel_id)

        # Goal: assemble the car by placing the roof on the body and the wheels on the sides.
        # The target poses are calculated based on the body pose.
        roof_targ_pose = (body_pose[0] + np.array([0, 0, body_size[2] + roof_size[2]/2]), body_pose[1])
        wheel_targ_poses = [(body_pose[0] + np.array([0, body_size[1]/2 + wheel_size[1]/2, -body_size[2]/2]), body_pose[1]),
                            (body_pose[0] + np.array([0, -body_size[1]/2 - wheel_size[1]/2, -body_size[2]/2]), body_pose[1])]

        # Add the goals.
        self.add_goal(objs=[roof_id], matches=np.ones((1, 1)), targ_poses=[roof_targ_pose], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1/3, language_goal=self.lang_template)

        self.add_goal(objs=wheel_ids, matches=np.ones((2, 2)), targ_poses=wheel_targ_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=2/3, language_goal=self.lang_template)