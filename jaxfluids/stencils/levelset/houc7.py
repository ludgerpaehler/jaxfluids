from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative


class HOUC7(SpatialDerivative):
    def __init__(self, nh: int, inactive_axis: List):
        super(HOUC7, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self.coeff = [3.0, -28.0, 126.0, -420.0, 105.0, 252.0, -42.0, 4.0]

        self.sign = [1, -1]

        self._slices = [
            [
                [
                    jnp.s_[
                        ...,
                        jnp.s_[self.n - 4 * j : -self.n - 4 * j]
                        if -self.n - 4 * j != 0
                        else jnp.s_[self.n - 4 * j : None],
                        self.nhy,
                        self.nhz,
                    ],
                    jnp.s_[..., self.n - 3 * j : -self.n - 3 * j, self.nhy, self.nhz],
                    jnp.s_[..., self.n - 2 * j : -self.n - 2 * j, self.nhy, self.nhz],
                    jnp.s_[..., self.n - 1 * j : -self.n - 1 * j, self.nhy, self.nhz],
                    jnp.s_[..., self.n + 0 * j : -self.n + 0 * j, self.nhy, self.nhz],
                    jnp.s_[..., self.n + 1 * j : -self.n + 1 * j, self.nhy, self.nhz],
                    jnp.s_[..., self.n + 2 * j : -self.n + 2 * j, self.nhy, self.nhz],
                    jnp.s_[..., self.n + 3 * j : -self.n + 3 * j, self.nhy, self.nhz],
                ],
                [
                    jnp.s_[
                        ...,
                        self.nhx,
                        jnp.s_[self.n - 4 * j : -self.n - 4 * j]
                        if -self.n - 4 * j != 0
                        else jnp.s_[self.n - 4 * j : None],
                        self.nhz,
                    ],
                    jnp.s_[..., self.nhx, self.n - 3 * j : -self.n - 3 * j, self.nhz],
                    jnp.s_[..., self.nhx, self.n - 2 * j : -self.n - 2 * j, self.nhz],
                    jnp.s_[..., self.nhx, self.n - 1 * j : -self.n - 1 * j, self.nhz],
                    jnp.s_[..., self.nhx, self.n + 0 * j : -self.n + 0 * j, self.nhz],
                    jnp.s_[..., self.nhx, self.n + 1 * j : -self.n + 1 * j, self.nhz],
                    jnp.s_[..., self.nhx, self.n + 2 * j : -self.n + 2 * j, self.nhz],
                    jnp.s_[..., self.nhx, self.n + 3 * j : -self.n + 3 * j, self.nhz],
                ],
                [
                    jnp.s_[
                        ...,
                        self.nhx,
                        self.nhy,
                        jnp.s_[self.n - 4 * j : -self.n - 4 * j]
                        if -self.n - 4 * j != 0
                        else jnp.s_[self.n - 4 * j : None],
                    ],
                    jnp.s_[..., self.nhx, self.nhy, self.n - 3 * j : -self.n - 3 * j],
                    jnp.s_[..., self.nhx, self.nhy, self.n - 2 * j : -self.n - 2 * j],
                    jnp.s_[..., self.nhx, self.nhy, self.n - 1 * j : -self.n - 1 * j],
                    jnp.s_[..., self.nhx, self.nhy, self.n + 0 * j : -self.n + 0 * j],
                    jnp.s_[..., self.nhx, self.nhy, self.n + 1 * j : -self.n + 1 * j],
                    jnp.s_[..., self.nhx, self.nhy, self.n + 2 * j : -self.n + 2 * j],
                    jnp.s_[..., self.nhx, self.nhy, self.n + 3 * j : -self.n + 3 * j],
                ],
            ]
            for j in self.sign
        ]

    def derivative_xi(
        self, levelset: jnp.DeviceArray, dxi: float, i: int, j: int, *args
    ) -> jnp.DeviceArray:
        s1_ = self._slices[j][i]

        cell_state_xi_j = sum(levelset[s1_[k]] * self.coeff[k] for k in range(len(self.coeff)))
        cell_state_xi_j *= self.sign[j] * 1.0 / 420.0
        cell_state_xi_j *= 1.0 / dxi

        return cell_state_xi_j
