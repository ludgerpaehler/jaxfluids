import types
from typing import List, Union

import jax.numpy as jnp

from jaxfluids.materials.material import Material
from jaxfluids.unit_handler import UnitHandler


class SafeIdealGas(Material):
    """Implements the safe ideal gas law, i.e., prevents division by zero and negative square roots"""

    def __init__(
        self,
        unit_handler: UnitHandler,
        dynamic_viscosity: Union[float, str, types.LambdaType],
        sutherland_parameters: List,
        bulk_viscosity: float,
        thermal_conductivity: Union[float, str, types.LambdaType],
        prandtl_number: float,
        specific_heat_ratio: float,
        specific_gas_constant: float,
        **kwargs,
    ) -> None:
        super().__init__(
            unit_handler,
            dynamic_viscosity,
            sutherland_parameters,
            bulk_viscosity,
            thermal_conductivity,
            prandtl_number,
        )

        self.gamma = specific_heat_ratio
        self.R = unit_handler.non_dimensionalize(specific_gas_constant, "specific_gas_constant")
        self.cp = self.gamma / (self.gamma - 1) * self.R

    def get_psi(self, p, rho):
        """See base class."""
        return jnp.maximum(p, self.eps) / rho

    def get_grueneisen(self, rho):
        """See base class."""
        return self.gamma - 1

    def get_speed_of_sound(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        """See base class."""
        return jnp.sqrt(self.gamma * jnp.maximum(p, self.eps) / jnp.maximum(rho, self.eps))

    def get_pressure(self, e: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        """See base class."""
        return (self.gamma - 1) * e * rho

    def get_temperature(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        """See base class."""
        return p / (rho * self.R + self.eps)

    def get_energy(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        """See base class."""
        return jnp.maximum(p, self.eps) / jnp.maximum(rho, self.eps) / (self.gamma - 1)

    def get_total_energy(
        self,
        p: jnp.DeviceArray,
        rho: jnp.DeviceArray,
        u: jnp.DeviceArray,
        v: jnp.DeviceArray,
        w: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        """See base class."""
        # Total energy per unit volume
        return jnp.maximum(p, self.eps) / (self.gamma - 1) + 0.5 * jnp.maximum(rho, self.eps) * (
            (u * u + v * v + w * w)
        )

    def get_total_enthalpy(
        self,
        p: jnp.DeviceArray,
        rho: jnp.DeviceArray,
        u: jnp.DeviceArray,
        v: jnp.DeviceArray,
        w: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        """See base class."""
        # Total specific enthalpy
        return (self.get_total_energy(p, rho, u, v, w) + jnp.maximum(p, self.eps)) / jnp.maximum(
            rho, self.eps
        )
