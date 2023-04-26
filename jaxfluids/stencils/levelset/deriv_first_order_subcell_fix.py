from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative


class DerivativeFirstOrderSidedSubcellFix(SpatialDerivative):
    def __init__(self, nh: int, inactive_axis: List, offset: int = 0):
        super(DerivativeFirstOrderSidedSubcellFix, self).__init__(nh, inactive_axis, offset)

        self.s_ = [
            [
                [
                    jnp.s_[..., self.n - 1 + j : -self.n - 1 + j, self.nhy, self.nhz],
                    jnp.s_[
                        ...,
                        jnp.s_[self.n - 0 + j : -self.n - 0 + j]
                        if -self.n - 0 + j != 0
                        else jnp.s_[self.n - 0 + j : None],
                        self.nhy,
                        self.nhz,
                    ],
                ],
                [
                    jnp.s_[..., self.nhx, self.n - 1 + j : -self.n - 1 + j, self.nhz],
                    jnp.s_[
                        ...,
                        self.nhx,
                        jnp.s_[self.n - 0 + j : -self.n - 0 + j]
                        if -self.n - 0 + j != 0
                        else jnp.s_[self.n - 0 + j : None],
                        self.nhz,
                    ],
                ],
                [
                    jnp.s_[..., self.nhx, self.nhy, self.n - 1 + j : -self.n - 1 + j],
                    jnp.s_[
                        ...,
                        self.nhx,
                        self.nhy,
                        jnp.s_[self.n - 0 + j : -self.n - 0 + j]
                        if -self.n - 0 + j != 0
                        else jnp.s_[self.n - 0 + j : None],
                    ],
                ],
            ]
            for j in [0, 1]
        ]

        self.mask_indices = [
            [
                [
                    jnp.s_[self.nhx, self.nhy, self.nhz],
                    jnp.s_[self.n - 1 + j : -self.n - 1 + j, self.nhy, self.nhz],
                ],
                [
                    jnp.s_[self.nhx, self.nhy, self.nhz],
                    jnp.s_[self.nhx, self.n - 1 + j : -self.n - 1 + j, self.nhz],
                ],
                [
                    jnp.s_[self.nhx, self.nhy, self.nhz],
                    jnp.s_[self.nhx, self.nhy, self.n - 1 + j : -self.n - 1 + j],
                ],
            ]
            for j in [0, 2]
        ]

        self.sign = [1, -1]

    def derivative_xi(
        self,
        levelset: jnp.DeviceArray,
        dxi: jnp.DeviceArray,
        i: int,
        j: int,
        levelset_0: jnp.DeviceArray,
        distance: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        slice = self.s_[j][i]

        indices_mask = self.mask_indices[j][i]

        mask = jnp.where(levelset_0[indices_mask[0]] * levelset_0[indices_mask[1]] < 0, 1, 0)

        deriv_xi_interface = (
            self.sign[j]
            * levelset[..., self.nhx, self.nhy, self.nhz]
            / (jnp.abs(distance) + jnp.finfo(jnp.float64).eps)
        )

        deriv_xi = (1.0 / dxi) * (-levelset[slice[0]] + levelset[slice[1]])

        deriv_xi = mask * deriv_xi_interface + (1.0 - mask) * deriv_xi

        return deriv_xi
