from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative


class DerivativeFirstOrderSided(SpatialDerivative):
    def __init__(self, nh: int, inactive_axis: List, offset: int = 0):
        super(DerivativeFirstOrderSided, self).__init__(nh, inactive_axis, offset)

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

    def derivative_xi(
        self, levelset: jnp.DeviceArray, dxi: jnp.DeviceArray, i: int, j: int, *args
    ) -> jnp.DeviceArray:
        s1_ = self.s_[j][i]
        deriv_xi = (1.0 / dxi) * (-levelset[s1_[0]] + levelset[s1_[1]])
        return deriv_xi
