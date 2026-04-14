from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.env_warehouse import EnvWarehouse
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes

class EnvWarehouseExtraObjects(EnvWarehouse):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_extra_list = [
            MultiBoxField(
                np.array(
                    [
                        [0.85, 0.1, 0.25 / 2],
                        [0.6, -0.15, 0.5 / 2],
                    ]
                ),
                np.array(
                    [
                        [0.1, 0.25, 0.25],
                        [0.25, 0.25, 0.5],
                    ]
                ),
                tensor_args=tensor_args,
            )
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "tableshelf-extraobjects")], 
            tensor_args=tensor_args, 
            **kwargs
        )

if __name__ == "__main__":
    env = EnvWarehouseExtraObjects(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
