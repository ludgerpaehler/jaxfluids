from typing import Callable

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.utilities import get_fluxes_xi


class Rusanov(RiemannSolver):
    """Rusanov (Local Lax-Friedrichs) Riemann Solver"""

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

        alpha = jnp.maximum(
            jnp.abs(primes_L[axis + 1]) + speed_of_sound_left,
            jnp.abs(primes_R[axis + 1]) + speed_of_sound_right,
        )

        fluxes_xi = 0.5 * (fluxes_left + fluxes_right) - 0.5 * alpha * (cons_R - cons_L)

        return fluxes_xi
