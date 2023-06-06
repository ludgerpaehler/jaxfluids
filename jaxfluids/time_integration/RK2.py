from typing import List

import jax.numpy as jnp

from jaxfluids.time_integration.time_integrator import TimeIntegrator


class RungeKutta2(TimeIntegrator):
    """2nd-order TVD RK2 scheme"""

    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(RungeKutta2, self).__init__(nh, inactive_axis)
        self.no_stages = 2
        self.timestep_multiplier = (1.0, 0.5)
        self.timestep_increment_factor = (1.0, 1.0)

    def prepare_buffer_for_integration(
        self, cons: jnp.DeviceArray, init: jnp.DeviceArray, stage: int
    ) -> jnp.DeviceArray:
        """u_cons = 0.5 u^n + 0.5 u^*"""
        return 0.5 * cons + 0.5 * init

    def integrate(
        self, cons: jnp.DeviceArray, rhs: jnp.DeviceArray, timestep: float, stage: int
    ) -> jnp.DeviceArray:
        timestep = timestep * self.timestep_multiplier[stage]
        cons = self.integrate_conservatives(cons, rhs, timestep)
        return cons
