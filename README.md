# JAX-Fluids: A Differentiable Fluid Dynamics Package

## About

JAX-Fluids is a fully-differentiable CFD solver for 3D, compressible two-phase flows.
We developed this package with the intention to push and facilitate research at the intersection
of ML and CFD. It is easy to use - running a simulation only requires a couple 
lines of code. Written entirely in JAX, the solver runs on CPU/GPU/TPU and 
enables automatic differentiation for end-to-end optimization 
of numerical models.

To learn more about implementation details and details on numerical methods provided 
by Jaxfluids, feel free to read [our paper](https://www.sciencedirect.com/science/article/abs/pii/S0010465522002466).

Authors:

- [Deniz A. Bezgin](https://www.epc.ed.tum.de/en/aer/mitarbeiter-innen/cv-2/a-d/m-sc-deniz-bezgin/)
- [Aaron B. Buhendwa](https://www.epc.ed.tum.de/en/aer/mitarbeiter-innen/cv-2/a-d/m-sc-aaron-buhendwa/)
- [Nikolaus A. Adams](https://www.epc.ed.tum.de/en/aer/members/cv/prof-adams/)

Correspondence via [mail](mailto:aaron.buhendwa@tum.de,mailto:deniz.bezgin@tum.de).

## Getting Started

To get started with Jaxfluids, we highly recommend to have a dive into the [JuPyter notebooks](./notebooks). All of which can be opened on [Google Colab](https://colab.research.google.com):

* SOD Shocktube: [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adopt-opt/jaxfluids/blob/main/notebooks/01_Sod/01_JAX-Fluids_1D_Sod_demo.ipynb)
* Bow Shock: [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adopt-opt/jaxfluids/blob/main/notebooks/02_Bowshock/02_JAX-Fluids_2D_Bow_Shock_demo.ipynb)
* Taylor-Green Vortex: [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adopt-opt/jaxfluids/blob/main/notebooks/03_TGV/03_JAX-Fluids_3D_Taylor_Green_Vortex_demo.ipynb)
* How to Set Up a Case: [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adopt-opt/jaxfluids/blob/main/notebooks/04_Case_setup/04_JAX-Fluids_Case_Setup_demo.ipynb)
* How to Change the Numerical Setup: [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adopt-opt/jaxfluids/blob/main/notebooks/05_Numerical_setup/05_JAX-Fluids_Numerical_Setup_demo.ipynb)
* Automatic Differentiation in Jaxfluids: [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adopt-opt/jaxfluids/blob/main/notebooks/06_Automatic_differentiation/06_JAX-Fluids_Automatic_Differentiation.ipynb)
* Neural Networks in Jaxfluids: [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adopt-opt/jaxfluids/blob/main/notebooks/07_RusanovNN/07_JAX-Fluids_RusanovNN.ipynb)
* Cylinder Flow: [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adopt-opt/jaxfluids/blob/main/notebooks/08_Cylinderflow/08_Cylinderflow.ipynb)
* Laminar Channel Flow: [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adopt-opt/jaxfluids/blob/main/notebooks/09_Laminar_channelflow/09_Laminar_channelflow.ipynb)

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

## Quickstart
This github contains five [jupyter-notebooks](https://github.com/tumaer/JAXFLUIDS/tree/main/notebooks) which will get you started quickly.
They demonstrate how to run simple simulations like a 1D sod shock tube or 
a 2D supersonic cylinder flow. Furthermore, they show how you can easily
switch the numerical and/or case setup in order to, e.g., increase the order
of the spatial reconstruction stencil or decrease the resolution of the simulation.

## Upcoming features 
- 5-Equation diffuse interface model for multiphase flows 
- CPU/GPU/TPU parallelization based on homogenous domain decomposition

## Documentation
Check out the [documentation](https://jax-fluids.readthedocs.io/en/latest/index.html) of JAX-Fluids.

## Citation
https://doi.org/10.1016/j.cpc.2022.108527

```
@article{BEZGIN2022108527,
   title = {JAX-Fluids: A fully-differentiable high-order computational fluid dynamics solver for compressible two-phase flows},
   journal = {Computer Physics Communications},
   pages = {108527},
   year = {2022},
   issn = {0010-4655},
   doi = {https://doi.org/10.1016/j.cpc.2022.108527},
   url = {https://www.sciencedirect.com/science/article/pii/S0010465522002466},
   author = {Deniz A. Bezgin and Aaron B. Buhendwa and Nikolaus A. Adams},
   keywords = {Computational fluid dynamics, Machine learning, Differential programming, Navier-Stokes equations, Level-set, Turbulence, Two-phase flows}
} 
```
## License
This project is licensed under the GNU General Public License v3 - see 
the [LICENSE](LICENSE) file or for details https://www.gnu.org/licenses/.
