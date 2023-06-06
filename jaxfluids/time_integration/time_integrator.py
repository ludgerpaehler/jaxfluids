from abc import ABC, abstractmethod
from functools import partial
from typing import List

import jax
import jax.numpy as jnp


class TimeIntegrator(ABC):
    """Abstract base class for explicit time integration schemes.
    All time intergration schemes are derived from TimeIntegrator.
    """

    def __init__(self, nh: int, inactive_axis: List) -> None:
        self.no_stages = None
        self.nhx = jnp.s_[:] if "x" in inactive_axis else jnp.s_[nh:-nh]
        self.nhy = jnp.s_[:] if "y" in inactive_axis else jnp.s_[nh:-nh]
        self.nhz = jnp.s_[:] if "z" in inactive_axis else jnp.s_[nh:-nh]

        self.timestep_multiplier = ()
        self.timestep_increment_factor = ()

    def integrate_conservatives(
        self, cons: jnp.DeviceArray, rhs: jnp.DeviceArray, timestep: float
    ) -> jnp.DeviceArray:
        """Integrates the conservative variables.

        :param cons: conservative variables buffer before integration
        :type cons: jnp.DeviceArray
        :param rhs: right-hand side buffer
        :type rhs: jnp.DeviceArray
        :param timestep: timestep adjusted according to sub-stage in Runge-Kutta
        :type timestep: float
        :return: conservative variables buffer after integration
        :rtype: DeviceArray
        """
        cons = cons.at[..., self.nhx, self.nhy, self.nhz].add(timestep * rhs)
        return cons

    @abstractmethod
    def integrate(
        self, cons: jnp.DeviceArray, rhs: jnp.DeviceArray, timestep: float, stage: int
    ) -> jnp.DeviceArray:
        """Wrapper function around integrate_conservatives. Adjusts the timestep
        according to current RK stage and calls integrate_conservatives.
        Implementation in child class.

        :param cons: conservative variables buffer before integration
        :type cons: jnp.DeviceArray
        :param rhs: right-hand side buffer
        :type rhs: jnp.DeviceArray
        :param timestep: timestep to be integrated
        :type timestep: float
        :return: conservative variables buffer after integration
        :rtype: DeviceArray
        """
        pass

    def prepare_buffer_for_integration(
        self, cons: jnp.DeviceArray, init: jnp.DeviceArray, stage: int
    ) -> jnp.DeviceArray:
        """In multi-stage Runge-Kutta methods, prepares the buffer for integration.
        Implementation in child class.

        :param cons: Buffer of conservative variables.
        :type cons: jnp.DeviceArray
        :param init: Initial conservative buffer.
        :type init: jnp.DeviceArray
        :param stage: Current stage of the RK time integrator.
        :type stage: int
        :return: Sum of initial buffer and current buffer.
        :rtype: jnp.DeviceArray
        """
        pass
