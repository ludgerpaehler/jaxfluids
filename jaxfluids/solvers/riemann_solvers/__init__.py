from jaxfluids.solvers.riemann_solvers.HLL import HLL
from jaxfluids.solvers.riemann_solvers.HLLC import HLLC
from jaxfluids.solvers.riemann_solvers.HLLCLM import HLLCLM
from jaxfluids.solvers.riemann_solvers.Rusanov import Rusanov
from jaxfluids.solvers.riemann_solvers.RusanovNN import RusanovNN
from jaxfluids.solvers.riemann_solvers.signal_speeds import (
    signal_speed_Arithmetic, signal_speed_Davis, signal_speed_Davis_2,
    signal_speed_Einfeldt, signal_speed_Rusanov, signal_speed_Toro)

DICT_RIEMANN_SOLVER = {
    "HLL": HLL,
    "HLLC": HLLC,
    "HLLCLM": HLLCLM,
    "RUSANOV": Rusanov,
    "RUSANOVNN": RusanovNN,
}

DICT_SIGNAL_SPEEDS = {
    "ARITHMETIC": signal_speed_Arithmetic,
    "RUSANOV": signal_speed_Rusanov,
    "DAVIS": signal_speed_Davis,
    "DAVIS2": signal_speed_Davis_2,
    "EINFELDT": signal_speed_Einfeldt,
    "TORO": signal_speed_Toro,
}
