from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative


class SecondDerivativeSecondOrderCenter(SpatialDerivative):
    """
    2nd order stencil for 1st derivative at the cell center
            x
    |     |   |     |
    | i-1 | i | i+1 |
    |     |   |     |
    """

    def __init__(self, nh: int, inactive_axis: List, offset: int = 0) -> None:
        super(SecondDerivativeSecondOrderCenter, self).__init__(
            nh=nh, inactive_axis=inactive_axis, offset=offset
        )

        self.s_ = [
            [
                jnp.s_[..., self.n - 1 : -self.n - 1, self.nhy, self.nhz],  # i-1
                jnp.s_[..., self.n : -self.n, self.nhy, self.nhz],  # i
                jnp.s_[
                    ...,
                    jnp.s_[self.n + 1 : -self.n + 1] if self.n != 1 else jnp.s_[self.n + 1 : None],
                    self.nhy,
                    self.nhz,
                ],
            ],  # i+1
            [
                jnp.s_[..., self.nhx, self.n - 1 : -self.n - 1, self.nhz],
                jnp.s_[..., self.nhx, self.n : -self.n, self.nhz],
                jnp.s_[
                    ...,
                    self.nhx,
                    jnp.s_[self.n + 1 : -self.n + 1] if self.n != 1 else jnp.s_[self.n + 1 : None],
                    self.nhz,
                ],
            ],
            [
                jnp.s_[..., self.nhx, self.nhy, self.n - 1 : -self.n - 1],
                jnp.s_[..., self.nhx, self.nhy, self.n : -self.n],
                jnp.s_[
                    ...,
                    self.nhx,
                    self.nhy,
                    jnp.s_[self.n + 1 : -self.n + 1] if self.n != 1 else jnp.s_[self.n + 1 : None],
                ],
            ],
        ]

        # MIXED DERIVATIVE
        self.s__ = [
            [
                jnp.s_[
                    ..., self.n - 1 : -self.n - 1, self.n - 1 : -self.n - 1, self.nhz
                ],  # i-1,j-1,k
                jnp.s_[
                    ..., self.n + 1 : -self.n + 1, self.n - 1 : -self.n - 1, self.nhz
                ],  # i+1,j-1,k
                jnp.s_[
                    ..., self.n - 1 : -self.n - 1, self.n + 1 : -self.n + 1, self.nhz
                ],  # i-1,j+1,k
                jnp.s_[..., self.n + 1 : -self.n + 1, self.n + 1 : -self.n + 1, self.nhz],
            ],  # i+1,j+1,k
            [
                jnp.s_[
                    ..., self.n - 1 : -self.n - 1, self.nhy, self.n - 1 : -self.n - 1
                ],  # i-1,j,k-1
                jnp.s_[
                    ..., self.n + 1 : -self.n + 1, self.nhy, self.n - 1 : -self.n - 1
                ],  # i+1,j,k-1
                jnp.s_[
                    ..., self.n - 1 : -self.n - 1, self.nhy, self.n + 1 : -self.n + 1
                ],  # i-1,j,k+1
                jnp.s_[..., self.n + 1 : -self.n + 1, self.nhy, self.n + 1 : -self.n + 1],
            ],  # i+1,j,k+1
            [
                jnp.s_[
                    ..., self.nhx, self.n - 1 : -self.n - 1, self.n - 1 : -self.n - 1
                ],  # i,j-1,k-1
                jnp.s_[
                    ..., self.nhx, self.n + 1 : -self.n + 1, self.n - 1 : -self.n - 1
                ],  # i,j+1,k-1
                jnp.s_[
                    ..., self.nhx, self.n - 1 : -self.n - 1, self.n + 1 : -self.n + 1
                ],  # i,j-1,k+1
                jnp.s_[..., self.nhx, self.n + 1 : -self.n + 1, self.n + 1 : -self.n + 1],
            ],  # i,j+1,k+1
        ]

        self.index_pair_dict = {"01": 0, "02": 1, "12": 2}

    def derivative_xi(
        self, buffer: jnp.DeviceArray, dxi: jnp.DeviceArray, i: int
    ) -> jnp.DeviceArray:
        s1_ = self.s_[i]
        deriv_xi = (1.0 / dxi / dxi) * (buffer[s1_[0]] - 2.0 * buffer[s1_[1]] + buffer[s1_[2]])
        return deriv_xi

    def derivative_xi_xj(
        self, buffer: jnp.DeviceArray, dxi: jnp.DeviceArray, dxj: jnp.DeviceArray, i: int, j: int
    ) -> jnp.DeviceArray:
        s1_ = self.s__[self.index_pair_dict[str(i) + (str(j))]]
        deriv_xi_xj = (
            1.0
            / 4.0
            / dxi
            / dxj
            * (buffer[s1_[0]] - buffer[s1_[1]] - buffer[s1_[2]] + buffer[s1_[3]])
        )
        return deriv_xi_xj
