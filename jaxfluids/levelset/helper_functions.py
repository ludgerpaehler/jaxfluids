import jax.numpy as jnp

# TODO MEMORY FOOTPRINT OF THESE FUNCTIONS IS HIGH - LOOPS ?


def move_source_to_target_ii(
    source_array: jnp.DeviceArray, normal_sign: jnp.DeviceArray, axis: int
) -> jnp.DeviceArray:
    """Moves the source array in positive normal direction within the ii plane.

    :param source_array: Source array buffer
    :type source_array: jnp.DeviceArray
    :param normal_sign: Normal sign buffer
    :type normal_sign: jnp.DeviceArray
    :param axis: axis i
    :type axis: int
    :return: Moved source array in ii plane
    :rtype: jnp.DeviceArray
    """
    array_plus = jnp.roll(source_array, 1, -3 + axis) * jnp.where(
        jnp.roll(normal_sign[axis], 1, -3 + axis) > 0, 1, 0
    )
    array_minus = jnp.roll(source_array, -1, -3 + axis) * jnp.where(
        jnp.roll(normal_sign[axis], -1, -3 + axis) < 0, 1, 0
    )
    array = array_plus + array_minus
    return array


def move_source_to_target_ij(
    source_array: jnp.DeviceArray, normal_sign: jnp.DeviceArray, axis_i: int, axis_j: int
) -> jnp.DeviceArray:
    normal_sign_i_plus_j_plus = jnp.roll(jnp.roll(normal_sign, 1, -3 + axis_i), 1, -3 + axis_j)
    normal_sign_i_plus_j_minus = jnp.roll(jnp.roll(normal_sign, 1, -3 + axis_i), -1, -3 + axis_j)
    normal_sign_i_minus_j_plus = jnp.roll(jnp.roll(normal_sign, -1, -3 + axis_i), 1, -3 + axis_j)
    normal_sign_i_minus_j_minus = jnp.roll(jnp.roll(normal_sign, -1, -3 + axis_i), -1, -3 + axis_j)
    array_i_plus_j_plus = jnp.roll(
        jnp.roll(source_array, 1, -3 + axis_i), 1, -3 + axis_j
    ) * jnp.where(
        (normal_sign_i_plus_j_plus[axis_i] > 0) & (normal_sign_i_plus_j_plus[axis_j] > 0), 1, 0
    )
    array_i_plus_j_minus = jnp.roll(
        jnp.roll(source_array, 1, -3 + axis_i), -1, -3 + axis_j
    ) * jnp.where(
        (normal_sign_i_plus_j_minus[axis_i] > 0) & (normal_sign_i_plus_j_minus[axis_j] < 0), 1, 0
    )
    array_i_minus_j_plus = jnp.roll(
        jnp.roll(source_array, -1, -3 + axis_i), 1, -3 + axis_j
    ) * jnp.where(
        (normal_sign_i_minus_j_plus[axis_i] < 0) & (normal_sign_i_minus_j_plus[axis_j] > 0), 1, 0
    )
    array_i_minus_j_minus = jnp.roll(
        jnp.roll(source_array, -1, -3 + axis_i), -1, -3 + axis_j
    ) * jnp.where(
        (normal_sign_i_minus_j_minus[axis_i] < 0) & (normal_sign_i_minus_j_minus[axis_j] < 0), 1, 0
    )
    array = (
        array_i_plus_j_plus + array_i_plus_j_minus + array_i_minus_j_plus + array_i_minus_j_minus
    )
    return array


def move_source_to_target_ijk(
    source_array: jnp.DeviceArray, normal_sign: jnp.DeviceArray
) -> jnp.DeviceArray:
    normal_sign_i_plus_j_plus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(normal_sign, 1, -3), 1, -2), 1, -1
    )
    normal_sign_i_plus_j_minus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(normal_sign, 1, -3), -1, -2), 1, -1
    )
    normal_sign_i_minus_j_plus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(normal_sign, -1, -3), 1, -2), 1, -1
    )
    normal_sign_i_minus_j_minus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(normal_sign, -1, -3), -1, -2), 1, -1
    )
    normal_sign_i_plus_j_plus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(normal_sign, 1, -3), 1, -2), -1, -1
    )
    normal_sign_i_plus_j_minus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(normal_sign, 1, -3), -1, -2), -1, -1
    )
    normal_sign_i_minus_j_plus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(normal_sign, -1, -3), 1, -2), -1, -1
    )
    normal_sign_i_minus_j_minus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(normal_sign, -1, -3), -1, -2), -1, -1
    )
    array_i_plus_j_plus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(source_array, 1, -3), 1, -2), 1, -1
    ) * jnp.where(
        (normal_sign_i_plus_j_plus_k_plus[0] > 0)
        & (normal_sign_i_plus_j_plus_k_plus[1] > 0)
        & (normal_sign_i_plus_j_plus_k_plus[2] > 0),
        1,
        0,
    )
    array_i_plus_j_minus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(source_array, 1, -3), -1, -2), 1, -1
    ) * jnp.where(
        (normal_sign_i_plus_j_minus_k_plus[0] > 0)
        & (normal_sign_i_plus_j_minus_k_plus[1] < 0)
        & (normal_sign_i_plus_j_minus_k_plus[2] > 0),
        1,
        0,
    )
    array_i_minus_j_plus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(source_array, -1, -3), 1, -2), 1, -1
    ) * jnp.where(
        (normal_sign_i_minus_j_plus_k_plus[0] < 0)
        & (normal_sign_i_minus_j_plus_k_plus[1] > 0)
        & (normal_sign_i_minus_j_plus_k_plus[2] > 0),
        1,
        0,
    )
    array_i_minus_j_minus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(source_array, -1, -3), -1, -2), 1, -1
    ) * jnp.where(
        (normal_sign_i_minus_j_minus_k_plus[0] < 0)
        & (normal_sign_i_minus_j_minus_k_plus[1] < 0)
        & (normal_sign_i_minus_j_minus_k_plus[2] > 0),
        1,
        0,
    )
    array_i_plus_j_plus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(source_array, 1, -3), 1, -2), -1, -1
    ) * jnp.where(
        (normal_sign_i_plus_j_plus_k_minus[0] > 0)
        & (normal_sign_i_plus_j_plus_k_minus[1] > 0)
        & (normal_sign_i_plus_j_plus_k_minus[2] < 0),
        1,
        0,
    )
    array_i_plus_j_minus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(source_array, 1, -3), -1, -2), -1, -1
    ) * jnp.where(
        (normal_sign_i_plus_j_minus_k_minus[0] > 0)
        & (normal_sign_i_plus_j_minus_k_minus[1] < 0)
        & (normal_sign_i_plus_j_minus_k_minus[2] < 0),
        1,
        0,
    )
    array_i_minus_j_plus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(source_array, -1, -3), 1, -2), -1, -1
    ) * jnp.where(
        (normal_sign_i_minus_j_plus_k_minus[0] < 0)
        & (normal_sign_i_minus_j_plus_k_minus[1] > 0)
        & (normal_sign_i_minus_j_plus_k_minus[2] < 0),
        1,
        0,
    )
    array_i_minus_j_minus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(source_array, -1, -3), -1, -2), -1, -1
    ) * jnp.where(
        (normal_sign_i_minus_j_minus_k_minus[0] < 0)
        & (normal_sign_i_minus_j_minus_k_minus[1] < 0)
        & (normal_sign_i_minus_j_minus_k_minus[2] < 0),
        1,
        0,
    )
    array = (
        array_i_plus_j_plus_k_plus
        + array_i_plus_j_minus_k_plus
        + array_i_minus_j_plus_k_plus
        + array_i_minus_j_minus_k_plus
        + array_i_plus_j_plus_k_minus
        + array_i_plus_j_minus_k_minus
        + array_i_minus_j_plus_k_minus
        + array_i_minus_j_minus_k_minus
    )
    return array


def move_target_to_source_ii(
    target_array: jnp.DeviceArray, normal_sign: jnp.DeviceArray, axis: int
) -> jnp.DeviceArray:
    """Moves the target array in negative normal direction in the ii plane.

    :param target_array: Target array buffer
    :type target_array: jnp.DeviceArray
    :param normal_sign: Normal sign buffer
    :type normal_sign: jnp.DeviceArray
    :param axis: axis i
    :type axis: int
    :return: Moved target array in ii plane
    :rtype: jnp.DeviceArray
    """
    array_plus = jnp.roll(target_array, 1, -3 + axis) * jnp.where(normal_sign[axis] < 0, 1, 0)
    array_minus = jnp.roll(target_array, -1, -3 + axis) * jnp.where(normal_sign[axis] > 0, 1, 0)
    array = array_plus + array_minus
    return array


def move_target_to_source_ij(
    target_array: jnp.DeviceArray, normal_sign: jnp.DeviceArray, axis_i: int, axis_j: int
) -> jnp.DeviceArray:
    array_i_plus_j_plus = jnp.roll(
        jnp.roll(target_array, 1, -3 + axis_i), 1, -3 + axis_j
    ) * jnp.where((normal_sign[axis_i] < 0) & (normal_sign[axis_j] < 0), 1, 0)
    array_i_plus_j_minus = jnp.roll(
        jnp.roll(target_array, 1, -3 + axis_i), -1, -3 + axis_j
    ) * jnp.where((normal_sign[axis_i] < 0) & (normal_sign[axis_j] > 0), 1, 0)
    array_i_minus_j_plus = jnp.roll(
        jnp.roll(target_array, -1, -3 + axis_i), 1, -3 + axis_j
    ) * jnp.where((normal_sign[axis_i] > 0) & (normal_sign[axis_j] < 0), 1, 0)
    array_i_minus_j_minus = jnp.roll(
        jnp.roll(target_array, -1, -3 + axis_i), -1, -3 + axis_j
    ) * jnp.where((normal_sign[axis_i] > 0) & (normal_sign[axis_j] > 0), 1, 0)
    array = (
        array_i_plus_j_plus + array_i_plus_j_minus + array_i_minus_j_plus + array_i_minus_j_minus
    )
    return array


def move_target_to_source_ijk(
    target_array: jnp.DeviceArray, normal_sign: jnp.DeviceArray
) -> jnp.DeviceArray:
    array_i_plus_j_plus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(target_array, 1, -3), 1, -2), 1, -1
    ) * jnp.where((normal_sign[0] < 0) & (normal_sign[1] < 0) & (normal_sign[2] < 0), 1, 0)
    array_i_plus_j_minus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(target_array, 1, -3), -1, -2), 1, -1
    ) * jnp.where((normal_sign[0] < 0) & (normal_sign[1] > 0) & (normal_sign[2] < 0), 1, 0)
    array_i_minus_j_plus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(target_array, -1, -3), 1, -2), 1, -1
    ) * jnp.where((normal_sign[0] > 0) & (normal_sign[1] < 0) & (normal_sign[2] < 0), 1, 0)
    array_i_minus_j_minus_k_plus = jnp.roll(
        jnp.roll(jnp.roll(target_array, -1, -3), -1, -2), 1, -1
    ) * jnp.where((normal_sign[0] > 0) & (normal_sign[1] > 0) & (normal_sign[2] < 0), 1, 0)
    array_i_plus_j_plus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(target_array, 1, -3), 1, -2), -1, -1
    ) * jnp.where((normal_sign[0] < 0) & (normal_sign[1] < 0) & (normal_sign[2] > 0), 1, 0)
    array_i_plus_j_minus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(target_array, 1, -3), -1, -2), -1, -1
    ) * jnp.where((normal_sign[0] < 0) & (normal_sign[1] > 0) & (normal_sign[2] > 0), 1, 0)
    array_i_minus_j_plus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(target_array, -1, -3), 1, -2), -1, -1
    ) * jnp.where((normal_sign[0] > 0) & (normal_sign[1] < 0) & (normal_sign[2] > 0), 1, 0)
    array_i_minus_j_minus_k_minus = jnp.roll(
        jnp.roll(jnp.roll(target_array, -1, -3), -1, -2), -1, -1
    ) * jnp.where((normal_sign[0] > 0) & (normal_sign[1] > 0) & (normal_sign[2] > 0), 1, 0)
    array = (
        array_i_plus_j_plus_k_plus
        + array_i_plus_j_minus_k_plus
        + array_i_minus_j_plus_k_plus
        + array_i_minus_j_minus_k_plus
        + array_i_plus_j_plus_k_minus
        + array_i_plus_j_minus_k_minus
        + array_i_minus_j_plus_k_minus
        + array_i_minus_j_minus_k_minus
    )
    return array
