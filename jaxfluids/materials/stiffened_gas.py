import types
from typing import List, Union

import jax.numpy as jnp

from jaxfluids.materials.material import Material
from jaxfluids.unit_handler import UnitHandler


class StiffenedGas(Material):
    """Implements the stiffened gas equation of state."""

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
        background_pressure: float,
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
        self.pb = unit_handler.non_dimensionalize(background_pressure, "pressure")

    def get_psi(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        return (p + self.gamma * self.pb) / rho

    def get_grueneisen(self, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        return self.gamma - 1

    def get_speed_of_sound(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        return jnp.sqrt(self.gamma * (p + self.pb) / (rho + self.eps))

    def get_pressure(self, e: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        return (self.gamma - 1) * e * rho - self.gamma * self.pb

    def get_temperature(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        return (p + self.pb) / (rho * self.R)

    def get_energy(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        # Specific internal energy
        return (p + self.gamma * self.pb) / ((rho + self.eps) * (self.gamma - 1))

    def get_total_energy(
        self,
        p: jnp.DeviceArray,
        rho: jnp.DeviceArray,
        u: jnp.DeviceArray,
        v: jnp.DeviceArray,
        w: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        # Total energy per unit volume
        return (p + self.gamma * self.pb) / (self.gamma - 1) + 0.5 * rho * ((u * u + v * v + w * w))

    def get_total_enthalpy(
        self,
        p: jnp.DeviceArray,
        rho: jnp.DeviceArray,
        u: jnp.DeviceArray,
        v: jnp.DeviceArray,
        w: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        # Total specific enthalpy
        return (self.get_total_energy(p, rho, u, v, w) + p) / rho
