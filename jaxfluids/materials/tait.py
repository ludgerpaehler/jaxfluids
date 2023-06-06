import types
from typing import List, Union

import jax.numpy as jnp

from jaxfluids.materials.material import Material
from jaxfluids.unit_handler import UnitHandler


class Tait(Material):
    """Implements the tait equation of state."""

    def __init__(
        self,
        unit_handler: UnitHandler,
        dynamic_viscosity: Union[float, str, types.LambdaType],
        sutherland_parameters: List,
        bulk_viscosity: float,
        thermal_conductivity: Union[float, str, types.LambdaType],
        prandtl_number: float,
        specific_gas_constant: float,
        specific_heat_ratio: float = 7.15,
        A_param: float = 1.00e5,
        B_param: float = 3.31e8,
        rho_0: float = 1.00e3,
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

        self.A_param = unit_handler.non_dimensionalize(A_param, "pressure")
        self.B_param = unit_handler.non_dimensionalize(B_param, "pressure")
        self.rho_0 = unit_handler.non_dimensionalize(rho_0, "density")

    def get_psi(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        return self.gamma * (p + self.B_param - self.A_param) / rho

    def get_grueneisen(self, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        return 0.0

    def get_speed_of_sound(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        return jnp.sqrt(self.gamma * (p + self.B_param - self.A_param) / (rho + self.eps))

    def get_pressure(self, e: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        return self.A_param - self.B_param + self.B_param * (rho / self.rho_0) ** self.gamma

    def get_temperature(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        # Temperature is not defined for Tait.
        return jnp.zeros_like(p)

    def get_energy(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        # Specific internal energy
        return (p + self.B_param - self.A_param) / (self.gamma * rho) + (
            self.B_param - self.A_param
        ) / rho

    def get_total_energy(
        self,
        p: jnp.DeviceArray,
        rho: jnp.DeviceArray,
        u: jnp.DeviceArray,
        v: jnp.DeviceArray,
        w: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        # Total energy per unit volume
        return (
            (p + self.B_param - self.A_param) / self.gamma + self.B_param - self.A_param
        ) + 0.5 * rho * ((u * u + v * v + w * w))

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
