from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction


class WENO3(SpatialReconstruction):
    """WENO3-JS"""

    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(WENO3, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self.dr = [1 / 3, 2 / 3]

        self.dr_ = [
            [1 / 3, 2 / 3],
            [2 / 3, 1 / 3],
        ]
        self.cr_ = [
            [[-0.5, 1.5], [0.5, 0.5]],
            [[0.5, 0.5], [1.5, -0.5]],
        ]

        self._stencil_size = 4

        self._slices = [
            [
                [
                    jnp.s_[..., self.n - 2 + j : -self.n - 1 + j, self.nhy, self.nhz],
                    jnp.s_[..., self.n - 1 + j : -self.n + j, self.nhy, self.nhz],
                    jnp.s_[..., self.n + j : -self.n + 1 + j, self.nhy, self.nhz],
                ],
                [
                    jnp.s_[..., self.nhx, self.n - 2 + j : -self.n - 1 + j, self.nhz],
                    jnp.s_[..., self.nhx, self.n - 1 + j : -self.n + j, self.nhz],
                    jnp.s_[..., self.nhx, self.n + j : -self.n + 1 + j, self.nhz],
                ],
                [
                    jnp.s_[
                        ...,
                        self.nhx,
                        self.nhy,
                        self.n - 2 + j : -self.n - 1 + j,
                    ],
                    jnp.s_[
                        ...,
                        self.nhx,
                        self.nhy,
                        self.n - 1 + j : -self.n + j,
                    ],
                    jnp.s_[
                        ...,
                        self.nhx,
                        self.nhy,
                        self.n + j : -self.n + 1 + j,
                    ],
                ],
            ]
            for j in range(2)
        ]

    def set_slices_stencil(self) -> None:
        self._slices = [
            [
                [
                    jnp.s_[..., 0 + j, None:None, None:None],
                    jnp.s_[..., 1 + j, None:None, None:None],
                    jnp.s_[..., 2 + j, None:None, None:None],
                ],
                [
                    jnp.s_[..., None:None, 0 + j, None:None],
                    jnp.s_[..., None:None, 1 + j, None:None],
                    jnp.s_[..., None:None, 2 + j, None:None],
                ],
                [
                    jnp.s_[..., None:None, None:None, 0 + j],
                    jnp.s_[..., None:None, None:None, 1 + j],
                    jnp.s_[..., None:None, None:None, 2 + j],
                ],
            ]
            for j in range(2)
        ]

    def reconstruct_xi(
        self, buffer: jnp.DeviceArray, axis: int, j: int, dx: float = None, **kwargs
    ) -> jnp.DeviceArray:
        s1_ = self._slices[j][axis]

        beta_0 = (buffer[s1_[1]] - buffer[s1_[0]]) * (buffer[s1_[1]] - buffer[s1_[0]])
        beta_1 = (buffer[s1_[2]] - buffer[s1_[1]]) * (buffer[s1_[2]] - buffer[s1_[1]])

        one_beta_0_sq = 1.0 / ((self.eps + beta_0) * (self.eps + beta_0))
        one_beta_1_sq = 1.0 / ((self.eps + beta_1) * (self.eps + beta_1))

        alpha_0 = self.dr_[j][0] * one_beta_0_sq
        alpha_1 = self.dr_[j][1] * one_beta_1_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1)

        omega_0 = alpha_0 * one_alpha
        omega_1 = alpha_1 * one_alpha

        p_0 = self.cr_[j][0][0] * buffer[s1_[0]] + self.cr_[j][0][1] * buffer[s1_[1]]
        p_1 = self.cr_[j][1][0] * buffer[s1_[1]] + self.cr_[j][1][1] * buffer[s1_[2]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1

        return cell_state_xi_j
