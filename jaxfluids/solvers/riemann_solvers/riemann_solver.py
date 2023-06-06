from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager


class RiemannSolver(ABC):
    """Abstract base class for Riemann solvers.

    RiemannSolver has two fundamental attributes: a material manager and a signal speed.
    The solve_riemann_problem_xi method solves the one-dimensional Riemann problem.
    """

    eps = jnp.finfo(jnp.float64).eps

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        self.material_manager = material_manager
        self.signal_speed = signal_speed

    @abstractmethod
    def solve_riemann_problem_xi(
        self,
        primes_L: jnp.DeviceArray,
        primes_R: jnp.DeviceArray,
        cons_L: jnp.DeviceArray,
        cons_R: jnp.DeviceArray,
        axis: int,
        **kwargs,
    ) -> jnp.DeviceArray:
        """Solves one-dimensional Riemann problem in the direction as specified
        by the axis argument.

        :param primes_L: primtive variable buffer left of cell face
        :type primes_L: jnp.DeviceArray
        :param primes_R: primtive variable buffer right of cell face
        :type primes_R: jnp.DeviceArray
        :param cons_L: conservative variable buffer left of cell face
        :type cons_L: jnp.DeviceArray
        :param cons_R: conservative variable buffer right of cell face
        :type cons_R: jnp.DeviceArray
        :param axis: Spatial direction along which Riemann problem is solved.
        :type axis: int
        :return: buffer of fluxes in xi direction
        :rtype: jnp.DeviceArray
        """
        pass
