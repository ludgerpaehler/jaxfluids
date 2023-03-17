# JAX-Fluids: A Differentiable Fluid Dynamics Package

JAX-Fluids is a fully-differentiable CFD solver for 3D, compressible two-phase flows.
We developed this package with the intention to push and facilitate research at the intersection
of ML and CFD. It is easy to use - running a simulation only requires a couple 
lines of code. Written entirely in JAX, the solver runs on CPU/GPU/TPU and 
enables automatic differentiation for end-to-end optimization 
of numerical models.

To learn more about implementation details and details on numerical methods provided 
by JAX-Fluids, feel free to read [our paper](https://www.sciencedirect.com/science/article/abs/pii/S0010465522002466).
And also check out the [documentation](https://jax-fluids.readthedocs.io/en/latest/index.html) of JAX-Fluids.

## Physical models and numerical methods

JAX-Fluids solves the Navier-Stokes-equations using the finite-volume-method on a Cartesian grid. 
The current version provides the following features:
- Explicit time stepping (Euler, RK2, RK3)
- High-order adaptive spatial reconstruction (WENO-3/5/7, WENO-CU6, WENO-3NN, TENO)
- Riemann solvers (Lax-Friedrichs, Rusanov, HLL, HLLC, Roe)
- Implicit turbulence sub-grid scale model ALDM
- Two-phase simulations via level-set method
- Immersed solid boundaries via level-set method
- Forcings for temperature, mass flow rate and kinetic energy spectrum
- Boundary conditions: Symmetry, Periodic, Wall, Dirichlet, Neumann
- CPU/GPU/TPU capability

## Pip Installation
Before installing JAX-Fluids, please ensure that you have
an updated and upgraded pip version.
### CPU-only support
To install the CPU-only version of JAX-Fluids, you can run
```bash
git clone https://github.com/tumaer/JAXFLUIDS.git
cd JAXFLUIDS
pip install .
```
Note: if you want to install JAX-Fluids in editable mode,
e.g., for code development on your local machine, run
```bash
pip install --editable .
```

Note: if you want to use jaxlib on a Mac with M1 chip, check the discussion [here](https://github.com/google/jax/issues/5501).
### GPU and CPU support
If you want to install JAX-Fluids with CPU and GPU support, you must
first install [CUDA](https://developer.nvidia.com/cuda-downloads) -
we have tested JAX-Fluids with CUDA 11.1 or newer.
After installing CUDA, run the following
```bash
git clone https://github.com/tumaer/JAXFLUIDS.git
cd JAXFLUIDS
pip install .[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For more information
on JAX on GPU please refer to the [github of JAX](https://github.com/google/jax)


## Documentation
Check out the [documentation](https://jax-fluids.readthedocs.io/en/latest/index.html) of JAX-Fluids.

## License
This project is licensed under the GNU General Public License v3 - see 
the [LICENSE](LICENSE) file or for details https://www.gnu.org/licenses/.
