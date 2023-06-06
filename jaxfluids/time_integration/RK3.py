from typing import List

import jax.numpy as jnp

from jaxfluids.time_integration.time_integrator import TimeIntegrator


class RungeKutta3(TimeIntegrator):
    """3rd-order TVD RK3 scheme"""

    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(RungeKutta3, self).__init__(nh, inactive_axis)
        self.no_stages = 3
        self.timestep_multiplier = (1.0, 0.25, 2.0 / 3.0)
        self.timestep_increment_factor = (1.0, 0.5, 1.0)
        self.conservatives_multiplier = [(0.25, 0.75), (2.0 / 3.0, 1.0 / 3.0)]

    def prepare_buffer_for_integration(
        self, cons: jnp.DeviceArray, init: jnp.DeviceArray, stage: int
    ) -> jnp.DeviceArray:
        """stage 1: u_cons = 3/4 u^n + 1/4 u^*
        stage 2: u_cons = 1/3 u^n + 2/3 u^**"""
        return (
            self.conservatives_multiplier[stage - 1][0] * cons
            + self.conservatives_multiplier[stage - 1][1] * init
        )

    def integrate(
        self, cons: jnp.DeviceArray, rhs: jnp.DeviceArray, timestep: float, stage: int
    ) -> jnp.DeviceArray:
        timestep = timestep * self.timestep_multiplier[stage]
        cons = self.integrate_conservatives(cons, rhs, timestep)
        return cons
