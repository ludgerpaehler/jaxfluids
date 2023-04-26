from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction


class WENO1(SpatialReconstruction):
    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(WENO1, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self._stencil_size = 2

        self._slices = [
            [
                [
                    jnp.s_[..., self.n - 1 + j : -self.n + j, self.nhy, self.nhz],
                ],
                [
                    jnp.s_[..., self.nhx, self.n - 1 + j : -self.n + j, self.nhz],
                ],
                [
                    jnp.s_[..., self.nhx, self.nhy, self.n - 1 + j : -self.n + j],
                ],
            ]
            for j in range(2)
        ]

    def set_slices_stencil(self) -> None:
        self._slices = [
            [
                [
                    jnp.s_[..., 0 + j, None:None, None:None],
                ],
                [
                    jnp.s_[..., None:None, 0 + j, None:None],
                ],
                [
                    jnp.s_[..., None:None, None:None, 0 + j],
                ],
            ]
            for j in range(2)
        ]

    def reconstruct_xi(
        self, buffer: jnp.DeviceArray, axis: int, j: int, dx=None, **kwargs
    ) -> jnp.DeviceArray:
        s1_ = self._slices[j][axis]

        cell_state_xi_j = buffer[s1_[0]]

        return cell_state_xi_j
