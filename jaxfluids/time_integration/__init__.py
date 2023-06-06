from jaxfluids.time_integration.euler import Euler
from jaxfluids.time_integration.RK2 import RungeKutta2
from jaxfluids.time_integration.RK3 import RungeKutta3

DICT_TIME_INTEGRATION = {
    "EULER": Euler,
    "RK2": RungeKutta2,
    "RK3": RungeKutta3,
}
