from typing import List

import jax.numpy as jnp

from jaxfluids.time_integration.time_integrator import TimeIntegrator


class Euler(TimeIntegrator):
    """First-order explicit Euler time integration scheme"""

    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(Euler, self).__init__(nh, inactive_axis)
        self.no_stages = 1
        self.timestep_multiplier = (1.0,)
        self.timestep_increment_factor = (1.0,)

    def integrate(
        self, cons: jnp.DeviceArray, rhs: jnp.DeviceArray, timestep: float, stage: int
    ) -> jnp.DeviceArray:
        cons = self.integrate_conservatives(cons, rhs, timestep)
        return cons
