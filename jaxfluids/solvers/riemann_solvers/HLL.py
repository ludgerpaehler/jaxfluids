from typing import Callable

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.utilities import get_fluxes_xi


class HLL(RiemannSolver):
    """HLL Riemann Solver by Harten, Lax and van Leer
    Harten et al. 1983
    """

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager, signal_speed)

    def solve_riemann_problem_xi(
        self,
        primes_L: jnp.DeviceArray,
        primes_R: jnp.DeviceArray,
        cons_L: jnp.DeviceArray,
        cons_R: jnp.DeviceArray,
        axis: int,
        **kwargs,
    ) -> jnp.DeviceArray:
        fluxes_left = get_fluxes_xi(primes_L, cons_L, axis)
        fluxes_right = get_fluxes_xi(primes_R, cons_R, axis)

        speed_of_sound_left = self.material_manager.get_speed_of_sound(
            p=primes_L[4], rho=primes_L[0]
        )
        speed_of_sound_right = self.material_manager.get_speed_of_sound(
            p=primes_R[4], rho=primes_R[0]
        )

        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(
            primes_L[axis + 1],
            primes_R[axis + 1],
            speed_of_sound_left,
            speed_of_sound_right,
            rho_L=primes_L[0],
            rho_R=primes_R[0],
            p_L=primes_L[4],
            p_R=primes_R[4],
            gamma=self.material_manager.gamma,
        )
        wave_speed_left = jnp.minimum(wave_speed_simple_L, 0.0)
        wave_speed_right = jnp.maximum(wave_speed_simple_R, 0.0)

        fluxes_xi = (
            wave_speed_right * fluxes_left
            - wave_speed_left * fluxes_right
            + wave_speed_left * wave_speed_right * (cons_R - cons_L)
        ) / (wave_speed_right - wave_speed_left + self.eps)

        return fluxes_xi
