from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction


class ALDM_WENO1(SpatialReconstruction):
    """ALDM_WENO1

    Implementation details provided in parent class.
    """

    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(ALDM_WENO1, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self._stencil_size = 6

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

    def reconstruct_xi(
        self, primes: jnp.DeviceArray, axis: int, j: int, dx: float = None, fs=0
    ) -> jnp.DeviceArray:
        s1_ = self._slices[j][axis]

        cell_state_xi_j = primes[s1_[0]]

        return cell_state_xi_j
