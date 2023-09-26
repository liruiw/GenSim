import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class BallOnBoxOnContainer(Task):
    """Pick up each ball and place it on the corresponding colored box, which are located in specific positions on a container."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "place the {} box on the container"
        self.lang_template_2 = "place the {} ball on the {} box"

        self.task_completed_desc = "done placing balls on boxs and box on container."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add container.
        container_size = (0.2, 0.2, 0.06)
        container_pose = self.get_random_pose(env, container_size)
        container_template = 'container/container-template.urdf'
        replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
        container_urdf = self.fill_template(container_template, replace)

        env.add_object(container_urdf, container_pose, 'fixed')

        # Define colors.
        ball_colors = ['red']
        box_colors = ['blue']

        # Add boxs.
        box_size = (0.04, 0.04, 0.06)
        box_template = 'box/box-template.urdf'
        boxs = []


        replace = {'DIM': box_size, 'HALF': (box_size[0] / 2, box_size[1] / 2, box_size[2] / 2), 'COLOR': ball_colors[0]}
        box_urdf = self.fill_template(box_template, replace)
        box_pose = self.get_random_pose(env, box_size)
        box_id = env.add_object(box_urdf, box_pose)
        boxs.append(box_id)

        # Add balls.
        ball_size = (0.04, 0.04, 0.04)
        ball_urdf = 'ball/ball.urdf'
        balls = []
        ball_pose = self.get_random_pose(env, ball_size)
        ball_id = env.add_object(ball_urdf, ball_pose, color=box_colors[0])
        balls.append(ball_id)

        # Goal: place the box on top of the container
        self.add_goal(objs=[boxs[0]], matches=np.ones((1, 1)), targ_poses=[container_pose], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1/2, language_goal=self.lang_template.format(box_colors[0]))


        # Goal: place the ball on top of the box
        language_goal = self.lang_template_2.format(ball_colors[0], box_colors[0])
        self.add_goal(objs=[balls[0]], matches=np.ones((1, 1)), targ_poses=[container_pose], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1/2, language_goal=language_goal)
