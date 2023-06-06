import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.shock_sensor.shock_sensor import ShockSensor
from jaxfluids.stencils.derivative.deriv_second_order_center import \
    DerivativeSecondOrderCenter


class Ducros(ShockSensor):
    """Ducros Shock Sensor

    fs = jnp.where(div / (div + curl + self.epsilon_s) >= 0.95, 1, 0)

    """

    def __init__(self, domain_information: DomainInformation) -> None:
        super().__init__(domain_information)

        self.derivative_stencil_center = DerivativeSecondOrderCenter(
            nh=domain_information.nh_conservatives,
            inactive_axis=domain_information.inactive_axis,
            offset=1,
        )

        self.n = 1
        self.shape_vel_grad = (
            3,
            domain_information.number_of_cells[0] + 2 * self.n,
            domain_information.number_of_cells[1] + 2 * self.n,
            domain_information.number_of_cells[2] + 2 * self.n,
        )
        self.nhx = (
            jnp.s_[:] if "x" in domain_information.inactive_axis else jnp.s_[self.n : -self.n]
        )
        self.nhy = (
            jnp.s_[:] if "y" in domain_information.inactive_axis else jnp.s_[self.n : -self.n]
        )
        self.nhz = (
            jnp.s_[:] if "z" in domain_information.inactive_axis else jnp.s_[self.n : -self.n]
        )

        self.s_ = [
            [jnp.s_[: -self.n, self.nhy, self.nhz], jnp.s_[self.n :, self.nhy, self.nhz]],
            [jnp.s_[self.nhx, : -self.n, self.nhz], jnp.s_[self.nhx, self.n :, self.nhz]],
            [jnp.s_[self.nhx, self.nhy, : -self.n], jnp.s_[self.nhx, self.nhy, self.n :]],
        ]

        self.epsilon_s = 1e-15

    def compute_sensor_function(self, vels: jnp.DeviceArray, axis: int) -> jnp.DeviceArray:
        if len(self.active_axis_indices) == 1:
            fs = 1.0
        else:
            vel_grad = self.compute_velocity_derivatives(vels)
            # EVALUATE DIV AND CURL AT CELL CENTER
            div = vel_grad[0, 0] + vel_grad[1, 1] + vel_grad[2, 2]
            curl_1 = vel_grad[1, 2] - vel_grad[2, 1]
            curl_2 = vel_grad[2, 0] - vel_grad[0, 2]
            curl_3 = vel_grad[0, 1] - vel_grad[1, 0]

            # CALCULATE DIV AND CURL AT CELL FACE
            div = div[self.s_[axis][0]] + div[self.s_[axis][1]]
            curl_1 = curl_1[self.s_[axis][0]] + curl_1[self.s_[axis][1]]
            curl_2 = curl_2[self.s_[axis][0]] + curl_2[self.s_[axis][1]]
            curl_3 = curl_3[self.s_[axis][0]] + curl_3[self.s_[axis][1]]

            div = jnp.abs(div)
            curl = jnp.sqrt(curl_1 * curl_1 + curl_2 * curl_2 + curl_3 * curl_3)

            fs = jnp.where(div / (div + curl + self.epsilon_s) >= 0.95, 1, 0)

        return fs

    def compute_velocity_derivatives(self, vels: jnp.DeviceArray) -> jnp.DeviceArray:
        """Computes the velocity gradient.
        Note that velocity gradients and especially curl and divergence
        are often used to determine the presence of shocks.

        vel_grad = [ du/dx dv/dx dw/dx
                     du/dy dv/dy dw/dy
                     du/dz dv/dz dw/dz ]

        :param vels: Buffer of velocities.
        :type vels: jnp.DeviceArray
        :return: Buffer of the velocity gradient.
        :rtype: jnp.DeviceArray
        """
        vel_grad = jnp.stack(
            [
                self.derivative_stencil_center.derivative_xi(vels, self.cell_sizes[i], i)
                if i in self.active_axis_indices
                else jnp.zeros(self.shape_vel_grad)
                for i in range(3)
            ]
        )
        return vel_grad
