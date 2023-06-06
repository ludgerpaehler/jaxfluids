from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction


class CentralSecondOrderReconstruction(SpatialReconstruction):
    """CentralSecondOrderReconstruction

    2nd order stencil for reconstruction at the cell face
        x
    |   |     |
    | i | i+1 |
    |   |     |
    """

    def __init__(self, nh: int, inactive_axis: List, offset: int) -> None:
        super(CentralSecondOrderReconstruction, self).__init__(
            nh=nh, inactive_axis=inactive_axis, offset=offset
        )

        self.s_ = [
            [
                jnp.s_[..., self.n - 1 : -self.n, self.nhy, self.nhz],  # i
                jnp.s_[
                    ...,
                    jnp.s_[self.n : -self.n + 1] if self.n != 1 else jnp.s_[self.n : None],
                    self.nhy,
                    self.nhz,
                ],
            ],  # i+1
            [
                jnp.s_[..., self.nhx, self.n - 1 : -self.n, self.nhz],
                jnp.s_[
                    ...,
                    self.nhx,
                    jnp.s_[self.n : -self.n + 1] if self.n != 1 else jnp.s_[self.n : None],
                    self.nhz,
                ],
            ],
            [
                jnp.s_[..., self.nhx, self.nhy, self.n - 1 : -self.n],
                jnp.s_[
                    ...,
                    self.nhx,
                    self.nhy,
                    jnp.s_[self.n : -self.n + 1] if self.n != 1 else jnp.s_[self.n : None],
                ],
            ],
        ]

    def reconstruct_xi(self, buffer: jnp.DeviceArray, axis: int, **kwargs) -> jnp.DeviceArray:
        s1_ = self.s_[axis]
        cell_state_xi = (1.0 / 2.0) * (buffer[s1_[0]] + buffer[s1_[1]])
        return cell_state_xi
