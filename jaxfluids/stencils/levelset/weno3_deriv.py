from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.levelset.deriv_first_order import \
    DerivativeFirstOrderSided
from jaxfluids.stencils.spatial_derivative import SpatialDerivative


class WENO3DERIV(SpatialDerivative):
    def __init__(self, nh: int, inactive_axis: List, face_value: str = "right"):
        offset = 2
        super(WENO3DERIV, self).__init__(offset, inactive_axis)
        self.derivative_stencil = DerivativeFirstOrderSided(nh, inactive_axis, offset)

        self.dr_ = [
            [1 / 3, 2 / 3],
            [2 / 3, 1 / 3],
        ]
        self.cr_ = [
            [[-0.5, 1.5], [0.5, 0.5]],
            [[0.5, 0.5], [1.5, -0.5]],
        ]

        self._slices = [
            [
                [
                    jnp.s_[..., self.n - 2 + j : -self.n - 1 + j, self.nhy, self.nhz],
                    jnp.s_[..., self.n - 1 + j : -self.n + 0 + j, self.nhy, self.nhz],
                    jnp.s_[
                        ...,
                        jnp.s_[self.n + 0 + j : -self.n + 1 + j]
                        if -self.n + 1 + j != 0
                        else jnp.s_[self.n + 0 + j : None],
                        self.nhy,
                        self.nhz,
                    ],
                ],
                [
                    jnp.s_[..., self.nhx, self.n - 2 + j : -self.n - 1 + j, self.nhz],
                    jnp.s_[..., self.nhx, self.n - 1 + j : -self.n + 0 + j, self.nhz],
                    jnp.s_[
                        ...,
                        self.nhx,
                        jnp.s_[self.n + 0 + j : -self.n + 1 + j]
                        if -self.n + 1 + j != 0
                        else jnp.s_[self.n + 0 + j : None],
                        self.nhz,
                    ],
                ],
                [
                    jnp.s_[..., self.nhx, self.nhy, self.n - 2 + j : -self.n - 1 + j],
                    jnp.s_[..., self.nhx, self.nhy, self.n - 1 + j : -self.n + 0 + j],
                    jnp.s_[
                        ...,
                        self.nhx,
                        self.nhy,
                        jnp.s_[self.n + 0 + j : -self.n + 1 + j]
                        if -self.n + 1 + j != 0
                        else jnp.s_[self.n + 0 + j : None],
                    ],
                ],
            ]
            for j in range(2)
        ]

        if face_value == "right":
            self.return_indices = [
                jnp.s_[..., 1:, :, :],
                jnp.s_[..., :, 1:, :],
                jnp.s_[..., :, :, 1:],
            ]
            self.sided_deriv_upwind = 0
        elif face_value == "left":
            self.return_indices = [
                jnp.s_[..., :-1, :, :],
                jnp.s_[..., :, :-1, :],
                jnp.s_[..., :, :, :-1],
            ]
            self.sided_deriv_upwind = 1
        else:
            assert False, "WENO3DERIV face_value must be left or right"

    def derivative_xi(
        self, levelset: jnp.DeviceArray, dx: float, i: int, j: int, *args
    ) -> jnp.DeviceArray:
        levelset = self.derivative_stencil.derivative_xi(levelset, dx, i, j=self.sided_deriv_upwind)
        s1_ = self._slices[j][i]

        beta_0 = (levelset[s1_[1]] - levelset[s1_[0]]) * (levelset[s1_[1]] - levelset[s1_[0]])
        beta_1 = (levelset[s1_[2]] - levelset[s1_[1]]) * (levelset[s1_[2]] - levelset[s1_[1]])

        one_beta_0_sq = 1.0 / ((self.eps + beta_0) * (self.eps + beta_0))
        one_beta_1_sq = 1.0 / ((self.eps + beta_1) * (self.eps + beta_1))

        alpha_0 = self.dr_[j][0] * one_beta_0_sq
        alpha_1 = self.dr_[j][1] * one_beta_1_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1)

        omega_0 = alpha_0 * one_alpha
        omega_1 = alpha_1 * one_alpha

        p_0 = self.cr_[j][0][0] * levelset[s1_[0]] + self.cr_[j][0][1] * levelset[s1_[1]]
        p_1 = self.cr_[j][1][0] * levelset[s1_[1]] + self.cr_[j][1][1] * levelset[s1_[2]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1

        return cell_state_xi_j[self.return_indices[i]]
