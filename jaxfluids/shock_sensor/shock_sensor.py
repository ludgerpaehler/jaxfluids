from abc import ABC, abstractmethod

from jaxfluids.domain_information import DomainInformation


class ShockSensor(ABC):
    """Abstract Class for shock sensors. Shock sensors indicate
    the presence of discontinuities by a marker.
    """

    def __init__(self, domain_information: DomainInformation) -> None:
        self.domain_information = domain_information
        self.cell_sizes = self.domain_information.cell_sizes
        self.active_axis_indices = [
            {"x": 0, "y": 1, "z": 2}[axis] for axis in domain_information.active_axis
        ]

    @abstractmethod
    def compute_sensor_function(self):
        """Computes the sensor function which is a marker (0/1)
        indicating the presence of shock discontinuities.

        Implementation in child classes.
        """
        pass
