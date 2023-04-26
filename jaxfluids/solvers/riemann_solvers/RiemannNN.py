from typing import Callable, Dict

import haiku
import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.utilities import get_fluxes_xi


class RiemannNN(RiemannSolver):
    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager=material_manager, signal_speed=signal_speed)

    def solve_riemann_problem_xi(
        self,
        primes_L: jnp.DeviceArray,
        primes_R: jnp.DeviceArray,
        cons_L: jnp.DeviceArray,
        cons_R: jnp.DeviceArray,
        axis: int,
        ml_params_dict: Dict,
        ml_networks_dict: Dict,
        **kwargs,
    ) -> jnp.DeviceArray:
        params = ml_params_dict["riemannsolver"]
        net = ml_networks_dict["riemannsolver"]

        fluxes_left = get_fluxes_xi(primes_L, cons_L, axis)
        fluxes_right = get_fluxes_xi(primes_R, cons_R, axis)

        speed_of_sound_left = self.material_manager.get_speed_of_sound(
            p=primes_L[4], rho=primes_L[0]
        )
        speed_of_sound_right = self.material_manager.get_speed_of_sound(
            p=primes_R[4], rho=primes_R[0]
        )
        speed_of_sound = 0.5 * (speed_of_sound_left + speed_of_sound_right)

        # STANDARD RUSANOV DISSIPATION
        alpha = jnp.maximum(
            jnp.abs(primes_L[axis + 1]) + speed_of_sound_left,
            jnp.abs(primes_R[axis + 1]) + speed_of_sound_right,
        )

        delta_vel = jnp.abs(primes_R[axis + 1] - primes_L[axis + 1])
        mean_vel = 0.5 * (primes_L[axis + 1] + primes_R[axis + 1])
        delta_mach = delta_vel / speed_of_sound
        mean_mach = jnp.abs(mean_vel) / speed_of_sound

        entropy_L = primes_L[4] / (primes_L[0]) ** self.material_manager.gamma
        entropy_R = primes_R[4] / (primes_R[0]) ** self.material_manager.gamma
        sum_entropy = entropy_L + entropy_R
        delta_entropy = jnp.abs(entropy_R - entropy_L)
        entropy_ratio = delta_entropy / sum_entropy

        # EVALUATION OF NEURAL NETWORK

        vec = jnp.stack([delta_mach, mean_mach, entropy_ratio])
        dissipation_nn = speed_of_sound * net.apply(params, vec)
        dissipation = jnp.minimum(dissipation_nn, alpha)
        # print(dissipation.shape)
        # exit()
        fluxes_xi = 0.5 * (fluxes_left + fluxes_right) - 0.5 * dissipation * (cons_R - cons_L)
        # fluxes_xi = 0.5 * (fluxes_left + fluxes_right) - jnp.einsum("ij...,j...->i...", dissipation, delta_u)
        # print(fluxes_xi.shape)
        # exit()
        return fluxes_xi
