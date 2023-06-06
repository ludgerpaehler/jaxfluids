from abc import ABC, abstractmethod
from typing import List

import jax.numpy as jnp


class SpatialReconstruction(ABC):
    """This is an abstract spatial reconstruction class. SpatialReconstruction
    class implements functionality for cell face reconstruction from cell
    averaged values. The paranet class implements the domain slices (nhx, nhy, nhz).
    The reconstruction procedure is implemented in the child classes.
    """

    eps = jnp.finfo(jnp.float64).eps

    def __init__(self, nh: int, inactive_axis: List, offset: int = 0) -> None:
        self.n = nh - offset
        self.nhx = jnp.s_[:] if "x" in inactive_axis else jnp.s_[self.n : -self.n]
        self.nhy = jnp.s_[:] if "y" in inactive_axis else jnp.s_[self.n : -self.n]
        self.nhz = jnp.s_[:] if "z" in inactive_axis else jnp.s_[self.n : -self.n]

        self._stencil_size = None

    # @abstractmethod
    def set_slices_stencil(self) -> None:
        """Sets slice objects used in eigendecomposition for flux-splitting scheme.
        In the flux-splitting scheme, each n-point stencil has to be separately
        accessible as each stencil is transformed into characteristic space.
        """
        pass

    @abstractmethod
    def reconstruct_xi(
        self, buffer: jnp.DeviceArray, axis: int, j: int, dx: float = None, **kwargs
    ) -> jnp.DeviceArray:
        """Reconstruction of buffer quantity along axis specified by axis.

        :param buffer: Buffer that will be reconstructed
        :type buffer: jnp.DeviceArray
        :param axis: Spatial axis along which values are reconstructed
        :type axis: int
        :param j: integer which specifies whether to calculate reconstruction left (j=0) or right (j=1)
            of an interface
        :type j: int
        :param dx: cell size, defaults to None
        :type dx: float, optional
        :return: Buffer with cell face reconstructed values
        :rtype: jnp.DeviceArray
        """
        pass
