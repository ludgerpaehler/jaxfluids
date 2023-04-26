from typing import Dict

import jax.numpy as jnp

from jaxfluids.materials import DICT_MATERIAL
from jaxfluids.materials.material import Material
from jaxfluids.unit_handler import UnitHandler


class MaterialManager:
    """The MaterialManager class is a wrapper class that holds the materials. The main purpose of this class is to enable
    the computation of material parameters for two-phase flows, i.e., the presence of two (different) materials.
    """

    eps = jnp.finfo(jnp.float64).eps

    def __init__(
        self, unit_handler: UnitHandler, material_properties: Dict, levelset_type: str
    ) -> None:
        self.levelset_type = levelset_type

        if levelset_type == "FLUID-FLUID":
            self.materials: Dict[str, Material] = {}

            for fluid in ["positive", "negative"]:
                self.materials[fluid] = DICT_MATERIAL[material_properties[fluid]["type"]](
                    unit_handler, **material_properties[fluid]
                )

            self.bulk_viscosity = jnp.stack(
                [
                    self.materials["positive"].bulk_viscosity,
                    self.materials["negative"].bulk_viscosity,
                ],
                axis=0,
            ).reshape(2, 1, 1, 1)
            self.gamma = jnp.stack(
                [self.materials["positive"].gamma, self.materials["negative"].gamma]
            ).reshape(2, 1, 1, 1)
            self.R = jnp.stack(
                [self.materials["positive"].R, self.materials["negative"].R]
            ).reshape(2, 1, 1, 1)
            self.sigma = unit_handler.non_dimensionalize(
                material_properties["pairing"]["surface_tension_coefficient"],
                "surface_tension_coefficient",
            )

        else:
            self.material: Material = DICT_MATERIAL[material_properties["type"]](
                unit_handler, **material_properties
            )
            self.bulk_viscosity = self.material.bulk_viscosity
            self.gamma = self.material.gamma
            self.R = self.material.R

    def get_thermal_conductivity(self, T: jnp.DeviceArray):
        if self.levelset_type == "FLUID-FLUID":
            thermal_conductivity_1 = self.materials["positive"].get_thermal_conductivity(T[0])
            thermal_conductivity_2 = self.materials["negative"].get_thermal_conductivity(T[1])
            thermal_conductivity = jnp.stack(
                [
                    thermal_conductivity_1 * jnp.ones_like(thermal_conductivity_2),
                    thermal_conductivity_2 * jnp.ones_like(thermal_conductivity_1),
                ],
                axis=0,
            )
            if thermal_conductivity.shape == (2,):
                thermal_conductivity = thermal_conductivity.reshape(2, 1, 1, 1)
        else:
            thermal_conductivity = self.material.get_thermal_conductivity(T)
        return thermal_conductivity

    def get_dynamic_viscosity(self, T: jnp.DeviceArray):
        if self.levelset_type == "FLUID-FLUID":
            dynamic_viscosity_1 = self.materials["positive"].get_dynamic_viscosity(T[0])
            dynamic_viscosity_2 = self.materials["negative"].get_dynamic_viscosity(T[1])
            dynamic_viscosity = jnp.stack(
                [
                    dynamic_viscosity_1 * jnp.ones_like(dynamic_viscosity_2),
                    dynamic_viscosity_2 * jnp.ones_like(dynamic_viscosity_1),
                ],
                axis=0,
            )
            if dynamic_viscosity.shape == (2,):
                dynamic_viscosity = dynamic_viscosity.reshape(2, 1, 1, 1)
        else:
            dynamic_viscosity = self.material.get_dynamic_viscosity(T)
        return dynamic_viscosity

    def get_speed_of_sound(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        if self.levelset_type == "FLUID-FLUID":
            speed_of_sound = []
            for i, fluid in enumerate(self.materials):
                speed_of_sound.append(self.materials[fluid].get_speed_of_sound(p[i], rho[i]))
            speed_of_sound = jnp.stack(speed_of_sound, axis=0)
        else:
            speed_of_sound = self.material.get_speed_of_sound(p, rho)
        return speed_of_sound

    def get_pressure(self, e: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        if self.levelset_type == "FLUID-FLUID":
            pressure = []
            for i, fluid in enumerate(self.materials):
                pressure.append(self.materials[fluid].get_pressure(e[i], rho[i]))
            pressure = jnp.stack(pressure, axis=0)
        else:
            pressure = self.material.get_pressure(e, rho)
        return pressure

    def get_temperature(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        if self.levelset_type == "FLUID-FLUID":
            temperature = []
            for i, fluid in enumerate(self.materials):
                temperature.append(self.materials[fluid].get_temperature(p[i], rho[i]))
            temperature = jnp.stack(temperature, axis=0)
        else:
            temperature = self.material.get_temperature(p, rho)
        return temperature

    def get_energy(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        # Specific internal energy
        if self.levelset_type == "FLUID-FLUID":
            energy = []
            for i, fluid in enumerate(self.materials):
                energy.append(self.materials[fluid].get_energy(p[i], rho[i]))
            energy = jnp.stack(energy, axis=0)
        else:
            energy = self.material.get_energy(p, rho)
        return energy

    def get_total_energy(
        self,
        p: jnp.DeviceArray,
        rho: jnp.DeviceArray,
        u: jnp.DeviceArray,
        v: jnp.DeviceArray,
        w: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        # Total energy per unit volume
        if self.levelset_type == "FLUID-FLUID":
            total_energy = []
            for i, fluid in enumerate(self.materials):
                total_energy.append(self.materials[fluid].get_total_energy(p[i], rho[i]))
            total_energy = jnp.stack(total_energy, axis=0)
        else:
            total_energy = self.material.get_total_energy(p, rho)
        return total_energy

    def get_total_enthalpy(
        self,
        p: jnp.DeviceArray,
        rho: jnp.DeviceArray,
        u: jnp.DeviceArray,
        v: jnp.DeviceArray,
        w: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        # Total specific enthalpy
        if self.levelset_type == "FLUID-FLUID":
            total_enthalpy = []
            for i, fluid in enumerate(self.materials):
                total_enthalpy.append(
                    self.materials[fluid].get_total_enthalpy(p[i], rho[i], u[i], v[i], w[i])
                )
            total_enthalpy = jnp.stack(total_enthalpy, axis=0)
        else:
            total_enthalpy = self.material.get_total_enthalpy(p, rho, u, v, w)
        return total_enthalpy

    def get_psi(self, p: jnp.DeviceArray, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        if self.levelset_type == "FLUID-FLUID":
            psi = []
            for i, fluid in enumerate(self.materials):
                psi.append(self.materials[fluid].get_psi(p[i], rho[i]))
            psi = jnp.stack(psi, axis=0)
        else:
            psi = self.material.get_psi(p, rho)
        return psi

    def get_grueneisen(self, rho: jnp.DeviceArray) -> jnp.DeviceArray:
        if self.levelset_type == "FLUID-FLUID":
            grueneisen = []
            for i, fluid in enumerate(self.materials):
                grueneisen.append(self.materials[fluid].get_grueneisen(rho[i]))
            grueneisen = jnp.stack(grueneisen, axis=0)
            if grueneisen.shape == (2,):
                grueneisen = grueneisen.reshape(2, 1, 1, 1)
        else:
            grueneisen = self.material.get_grueneisen(rho)
        return grueneisen
