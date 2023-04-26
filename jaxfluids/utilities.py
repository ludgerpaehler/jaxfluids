from functools import partial

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager


def get_conservatives_from_primitives(
    primes: jnp.DeviceArray, material_manager: MaterialManager
) -> jnp.DeviceArray:
    """Converts primitive variables to conservative variables.

    :param primes: Buffer of primitive variables
    :type primes: jnp.DeviceArray
    :param material_manager: Class that calculats material quantities
    :type material_manager: MaterialManager
    :return: Buffer of conservative variables
    :rtype: jnp.DeviceArray
    """
    e = material_manager.get_energy(p=primes[4], rho=primes[0])
    rho = primes[0]  # = rho
    rhou = primes[0] * primes[1]  # = rho * u
    rhov = primes[0] * primes[2]  # = rho * v
    rhow = primes[0] * primes[3]  # = rho * w
    E = primes[0] * (
        0.5 * (primes[1] * primes[1] + primes[2] * primes[2] + primes[3] * primes[3]) + e
    )  # E = rho * (1/2 u^2 + e)
    cons = jnp.stack([rho, rhou, rhov, rhow, E], axis=0)
    return cons


def get_primitives_from_conservatives(
    cons: jnp.DeviceArray, material_manager: MaterialManager
) -> jnp.DeviceArray:
    """Converts conservative variables to primitive variables.

    :param cons: Buffer of conservative variables
    :type cons: jnp.DeviceArray
    :param material_manager: Class that calculats material quantities
    :type material_manager: MaterialManager
    :return: Buffer of primitive variables
    :rtype: jnp.DeviceArray
    """
    rho = cons[0]  # rho = rho
    u = cons[1] / (cons[0] + jnp.finfo(float).eps)  # u = rho*u / rho
    v = cons[2] / (cons[0] + jnp.finfo(float).eps)  # v = rho*v / rho
    w = cons[3] / (cons[0] + jnp.finfo(float).eps)  # w = rho*w / rho
    e = cons[4] / (cons[0] + jnp.finfo(float).eps) - 0.5 * (u * u + v * v + w * w)
    p = material_manager.get_pressure(e, cons[0])  # p = (gamma-1) * ( E - 1/2 * (rho*u) * u)
    primes = jnp.stack([rho, u, v, w, p], axis=0)
    return primes


def get_fluxes_xi(primes: jnp.DeviceArray, cons: jnp.DeviceArray, axis: int) -> jnp.DeviceArray:
    """Computes the physical flux in a specified spatial direction.
    Cf. Eq. (3.65) in Toro.

    :param primes: Buffer of primitive variables
    :type primes: jnp.DeviceArray
    :param cons: Buffer of conservative variables
    :type cons: jnp.DeviceArray
    :param axis: Spatial direction along which fluxes are calculated
    :type axis: int
    :return: Physical fluxes in axis direction
    :rtype: jnp.DeviceArray
    """
    rho_ui = cons[axis + 1]  # (rho u_i)
    rho_ui_u1 = cons[axis + 1] * primes[1]  # (rho u_i) * u_1
    rho_ui_u2 = cons[axis + 1] * primes[2]  # (rho u_i) * u_2
    rho_ui_u3 = cons[axis + 1] * primes[3]  # (rho u_i) * u_3
    ui_Ep = primes[axis + 1] * (cons[4] + primes[4])
    if axis == 0:
        rho_ui_u1 += primes[4]
    elif axis == 1:
        rho_ui_u2 += primes[4]
    elif axis == 2:
        rho_ui_u3 += primes[4]
    flux_xi = jnp.stack([rho_ui, rho_ui_u1, rho_ui_u2, rho_ui_u3, ui_Ep], axis=0)
    return flux_xi
