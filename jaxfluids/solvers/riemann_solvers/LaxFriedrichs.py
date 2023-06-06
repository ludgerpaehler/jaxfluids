from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver


class LaxFriedrichs(RiemannSolver):
    def __init__(self, material_manager: MaterialManager, signal_speed) -> None:
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
        # TODO
        pass
