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
import pybullet as p

class ColorCoordinatedBallStacking(Task):
    """Stack balls on top of the corresponding colored containers in a specific color sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack the balls on top of the corresponding colored containers in the sequence blue, yellow, green, red"
        self.task_completed_desc = "done stacking balls."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the color sequence
        color_sequence = ['blue', 'yellow', 'green', 'red']

        # Add containers.
        container_size = (0.12, 0.12, 0.12)
        container_urdf = 'container/container-template.urdf'
        container_poses = []
        containers = []
        for color in color_sequence:
            container_pose = self.get_random_pose(env, container_size)
            container_id = env.add_object(container_urdf, container_pose, color=utils.COLORS[color])
            container_poses.append(container_pose)
            containers.append(container_id)

        # Add balls.
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball-template.urdf'
        balls = []
        for color in color_sequence:
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose, color=utils.COLORS[color])
            balls.append(ball_id)

        # Goal: each ball is stacked on top of the corresponding colored container in the color sequence.
        for i in range(len(balls)):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[container_poses[i]], replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1/len(balls),
                language_goal=self.lang_template.format(obj=color_sequence[i]))

        # Add distractors.
        n_distractors = 0
        while n_distractors < 6:
            is_ball = np.random.rand() > 0.5
            urdf = ball_urdf if is_ball else container_urdf
            size = ball_size if is_ball else container_size
            pose = self.get_random_pose(env, obj_size=size)
            color = np.random.choice(list(utils.COLORS.keys()))

            obj_id = env.add_object(urdf, pose, color=utils.COLORS[color])
            n_distractors += 1