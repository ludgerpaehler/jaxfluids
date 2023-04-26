from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction


class TENO5(SpatialReconstruction):
    """Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations"""

    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(TENO5, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self.dr_ = [
            [1 / 10, 6 / 10, 3 / 10],
            [3 / 10, 6 / 10, 1 / 10],
        ]
        self.cr_ = [
            [[1 / 3, -7 / 6, 11 / 6], [-1 / 6, 5 / 6, 1 / 3], [1 / 3, 5 / 6, -1 / 6]],
            [[-1 / 6, 5 / 6, 1 / 3], [1 / 3, 5 / 6, -1 / 6], [11 / 6, -7 / 6, 1 / 3]],
        ]

        self.C = 1.0
        self.q = 6
        self.CT = 1e-5

        self._stencil_size = 6

        self._slices = [
            [
                [
                    jnp.s_[..., self.n - 3 + j : -self.n - 2 + j, self.nhy, self.nhz],
                    jnp.s_[..., self.n - 2 + j : -self.n - 1 + j, self.nhy, self.nhz],
                    jnp.s_[..., self.n - 1 + j : -self.n + j, self.nhy, self.nhz],
                    jnp.s_[..., self.n + j : -self.n + 1 + j, self.nhy, self.nhz],
                    jnp.s_[..., self.n + 1 + j : -self.n + 2 + j, self.nhy, self.nhz],
                ],
                [
                    jnp.s_[..., self.nhx, self.n - 3 + j : -self.n - 2 + j, self.nhz],
                    jnp.s_[..., self.nhx, self.n - 2 + j : -self.n - 1 + j, self.nhz],
                    jnp.s_[..., self.nhx, self.n - 1 + j : -self.n + j, self.nhz],
                    jnp.s_[..., self.nhx, self.n + j : -self.n + 1 + j, self.nhz],
                    jnp.s_[..., self.nhx, self.n + 1 + j : -self.n + 2 + j, self.nhz],
                ],
                [
                    jnp.s_[..., self.nhx, self.nhy, self.n - 3 + j : -self.n - 2 + j],
                    jnp.s_[..., self.nhx, self.nhy, self.n - 2 + j : -self.n - 1 + j],
                    jnp.s_[..., self.nhx, self.nhy, self.n - 1 + j : -self.n + j],
                    jnp.s_[..., self.nhx, self.nhy, self.n + j : -self.n + 1 + j],
                    jnp.s_[..., self.nhx, self.nhy, self.n + 1 + j : -self.n + 2 + j],
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
                    jnp.s_[..., 3 + j, None:None, None:None],
                    jnp.s_[..., 4 + j, None:None, None:None],
                ],
                [
                    jnp.s_[..., None:None, 0 + j, None:None],
                    jnp.s_[..., None:None, 1 + j, None:None],
                    jnp.s_[..., None:None, 2 + j, None:None],
                    jnp.s_[..., None:None, 3 + j, None:None],
                    jnp.s_[..., None:None, 4 + j, None:None],
                ],
                [
                    jnp.s_[..., None:None, None:None, 0 + j],
                    jnp.s_[..., None:None, None:None, 1 + j],
                    jnp.s_[..., None:None, None:None, 2 + j],
                    jnp.s_[..., None:None, None:None, 3 + j],
                    jnp.s_[..., None:None, None:None, 4 + j],
                ],
            ]
            for j in range(2)
        ]

    def reconstruct_xi(
        self, buffer: jnp.DeviceArray, axis: int, j: int, dx: float = None, **kwargs
    ) -> jnp.DeviceArray:
        s1_ = self._slices[j][axis]

        beta_0 = 13.0 / 12.0 * (buffer[s1_[0]] - 2 * buffer[s1_[1]] + buffer[s1_[2]]) * (
            buffer[s1_[0]] - 2 * buffer[s1_[1]] + buffer[s1_[2]]
        ) + 1.0 / 4.0 * (buffer[s1_[0]] - 4 * buffer[s1_[1]] + 3 * buffer[s1_[2]]) * (
            buffer[s1_[0]] - 4 * buffer[s1_[1]] + 3 * buffer[s1_[2]]
        )
        beta_1 = 13.0 / 12.0 * (buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]) * (
            buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]
        ) + 1.0 / 4.0 * (buffer[s1_[1]] - buffer[s1_[3]]) * (buffer[s1_[1]] - buffer[s1_[3]])
        beta_2 = 13.0 / 12.0 * (buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]) * (
            buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]
        ) + 1.0 / 4.0 * (3 * buffer[s1_[2]] - 4 * buffer[s1_[3]] + buffer[s1_[4]]) * (
            3 * buffer[s1_[2]] - 4 * buffer[s1_[3]] + buffer[s1_[4]]
        )

        tau_5 = jnp.abs(beta_0 - beta_2)

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_5 / (beta_0 + self.eps)) ** self.q
        gamma_1 = (self.C + tau_5 / (beta_1 + self.eps)) ** self.q
        gamma_2 = (self.C + tau_5 / (beta_2 + self.eps)) ** self.q

        # gamma_0 *= (gamma_0 * gamma_0)
        # gamma_1 *= (gamma_1 * gamma_1)
        # gamma_2 *= (gamma_2 * gamma_2)

        # gamma_0 *= gamma_0
        # gamma_1 *= gamma_1
        # gamma_2 *= gamma_2

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2)

        # SHARP CUTOFF FUNCTION
        delta_0 = jnp.where(gamma_0 * one_gamma_sum < self.CT, 0.0, 1.0)
        delta_1 = jnp.where(gamma_1 * one_gamma_sum < self.CT, 0.0, 1.0)
        delta_2 = jnp.where(gamma_2 * one_gamma_sum < self.CT, 0.0, 1.0)

        w0 = delta_0 * self.dr_[j][0]
        w1 = delta_1 * self.dr_[j][1]
        w2 = delta_2 * self.dr_[j][2]

        one_dk = 1.0 / (w0 + w1 + w2)

        omega_0 = w0 * one_dk
        omega_1 = w1 * one_dk
        omega_2 = w2 * one_dk

        p_0 = (
            self.cr_[j][0][0] * buffer[s1_[0]]
            + self.cr_[j][0][1] * buffer[s1_[1]]
            + self.cr_[j][0][2] * buffer[s1_[2]]
        )
        p_1 = (
            self.cr_[j][1][0] * buffer[s1_[1]]
            + self.cr_[j][1][1] * buffer[s1_[2]]
            + self.cr_[j][1][2] * buffer[s1_[3]]
        )
        p_2 = (
            self.cr_[j][2][0] * buffer[s1_[2]]
            + self.cr_[j][2][1] * buffer[s1_[3]]
            + self.cr_[j][2][2] * buffer[s1_[4]]
        )

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2

        return cell_state_xi_j
