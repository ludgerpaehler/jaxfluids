from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative


class DerivativeFourthOrderCenter(SpatialDerivative):
    """
    4th order stencil for 1st derivative at the cell center
                  x
    |     |     |   |     |     |
    | i-2 | i-1 | i | i+1 | i+2 |
    |     |     |   |     |     |
    """

    def __init__(self, nh: int, inactive_axis: List, offset: int = 0) -> None:
        super(DerivativeFourthOrderCenter, self).__init__(
            nh=nh, inactive_axis=inactive_axis, offset=offset
        )

        self.s_ = [
            [
                jnp.s_[..., self.n - 2 : -self.n - 2, self.nhy, self.nhz],  # i-2
                jnp.s_[..., self.n - 1 : -self.n - 1, self.nhy, self.nhz],  # i-1
                jnp.s_[..., self.n + 1 : -self.n + 1, self.nhy, self.nhz],  # i+1
                jnp.s_[
                    ...,
                    jnp.s_[self.n + 2 : -self.n + 2] if self.n != 2 else jnp.s_[self.n + 2 : None],
                    self.nhy,
                    self.nhz,
                ],
            ],  # i+2
            [
                jnp.s_[..., self.nhx, self.n - 2 : -self.n - 2, self.nhz],
                jnp.s_[..., self.nhx, self.n - 1 : -self.n - 1, self.nhz],
                jnp.s_[..., self.nhx, self.n + 1 : -self.n + 1, self.nhz],
                jnp.s_[
                    ...,
                    self.nhx,
                    jnp.s_[self.n + 2 : -self.n + 2] if self.n != 2 else jnp.s_[self.n + 2 : None],
                    self.nhz,
                ],
            ],
            [
                jnp.s_[..., self.nhx, self.nhy, self.n - 2 : -self.n - 2],
                jnp.s_[..., self.nhx, self.nhy, self.n - 1 : -self.n - 1],
                jnp.s_[..., self.nhx, self.nhy, self.n + 1 : -self.n + 1],
                jnp.s_[
                    ...,
                    self.nhx,
                    self.nhy,
                    jnp.s_[self.n + 2 : -self.n + 2] if self.n != 2 else jnp.s_[self.n + 2 : None],
                ],
            ],
        ]

    def derivative_xi(
        self, buffer: jnp.DeviceArray, dxi: jnp.DeviceArray, axis: int
    ) -> jnp.DeviceArray:
        s1_ = self.s_[axis]
        deriv_xi = (1.0 / 12.0 / dxi) * (
            buffer[s1_[0]] - 8.0 * buffer[s1_[1]] + 8.0 * buffer[s1_[2]] - buffer[s1_[3]]
        )
        return deriv_xi
