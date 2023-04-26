from typing import Callable

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.solvers.riemann_solvers.signal_speeds import compute_sstar
from jaxfluids.utilities import get_fluxes_xi


class HLLC(RiemannSolver):
    """HLLC Riemann Solver
    Toro et al. 1994
    """

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager, signal_speed)
        self.s_star = compute_sstar

        # MINOR AXIS DIRECTIONS
        self.minor = [
            [2, 3],
            [3, 1],
            [1, 2],
        ]

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
        wave_speed_contact = self.s_star(
            primes_L[axis + 1],
            primes_R[axis + 1],
            primes_L[4],
            primes_R[4],
            primes_L[0],
            primes_R[0],
            wave_speed_simple_L,
            wave_speed_simple_R,
        )

        wave_speed_left = jnp.minimum(wave_speed_simple_L, 0.0)
        wave_speed_right = jnp.maximum(wave_speed_simple_R, 0.0)

        """ Toro 10.73 """
        pre_factor_L = (
            (wave_speed_simple_L - primes_L[axis + 1])
            / (wave_speed_simple_L - wave_speed_contact)
            * primes_L[0]
        )
        pre_factor_R = (
            (wave_speed_simple_R - primes_R[axis + 1])
            / (wave_speed_simple_R - wave_speed_contact)
            * primes_R[0]
        )

        u_star_L = [
            pre_factor_L,
            pre_factor_L,
            pre_factor_L,
            pre_factor_L,
            pre_factor_L
            * (
                cons_L[4] / cons_L[0]
                + (wave_speed_contact - primes_L[axis + 1])
                * (
                    wave_speed_contact
                    + primes_L[4] / primes_L[0] / (wave_speed_simple_L - primes_L[axis + 1])
                )
            ),
        ]
        u_star_L[axis + 1] *= wave_speed_contact
        u_star_L[self.minor[axis][0]] *= primes_L[self.minor[axis][0]]
        u_star_L[self.minor[axis][1]] *= primes_L[self.minor[axis][1]]
        u_star_L = jnp.stack(u_star_L)

        u_star_R = [
            pre_factor_R,
            pre_factor_R,
            pre_factor_R,
            pre_factor_R,
            pre_factor_R
            * (
                cons_R[4] / cons_R[0]
                + (wave_speed_contact - primes_R[axis + 1])
                * (
                    wave_speed_contact
                    + primes_R[4] / primes_R[0] / (wave_speed_simple_R - primes_R[axis + 1])
                )
            ),
        ]
        u_star_R[axis + 1] *= wave_speed_contact
        u_star_R[self.minor[axis][0]] *= primes_R[self.minor[axis][0]]
        u_star_R[self.minor[axis][1]] *= primes_R[self.minor[axis][1]]
        u_star_R = jnp.stack(u_star_R)

        """ Toro 10.72 """
        flux_star_L = fluxes_left + wave_speed_left * (u_star_L - cons_L)
        flux_star_R = fluxes_right + wave_speed_right * (u_star_R - cons_R)

        """ Kind of Toro 10.71 """
        fluxes_xi = (
            0.5 * (1 + jnp.sign(wave_speed_contact)) * flux_star_L
            + 0.5 * (1 - jnp.sign(wave_speed_contact)) * flux_star_R
        )
        return fluxes_xi
