# Kinodynamic Planning

Master's research: motion planning implementations for quadrotors (drones) as well as other robots with nonlinear dynamics.

## fmt (kino-FMT*)
The kino-FMT* algorithm presented here is my implementation of the asymptotically optimal and probabilistically complete **real-time** kinodynamic planning algorithm used specifically for quadrotors.

### The following demonstrations were generated using matplotlib and fmt_quad.py

#### Sampling and connecting
![sim_all_lines][sim_all_lines]

#### Selecting least-cost path and performing smoothing
![sim_smoothing][sim_smoothing]

[sim_all_lines]: https://github.com/luclarocque/kinodynamic-planning/blob/master/sim_all_lines.png "Different number of nodes sampled and connected with fmt"
[sim_smoothing]: https://github.com/luclarocque/kinodynamic-planning/blob/master/sim_smoothing.png "Least-cost path is found and smoothing operation is performed"

Based on work by [Ross Allen and Marco Pavone](https://stanfordasl.github.io/wp-content/papercite-data/pdf/Allen.Pavone.AIAAGNC16.pdf).


## sst (SST*)
The SST* algorithm presented here is my implementation of the asymptotically optimal and probabilistically complete (but not real-time) kinodynamic planning algorithm for arbitrary nonlinear systems, i.e., any robot! The real key behind this algorithm is its ability to use only forward integration to solve motion planning problems for systems with nonlinear dynamics. This is unlike many other solvers which require solutions to difficult/impossible boundary value problems.

### The following demonstrations were generated using pygame and sst_doubleint_parallel.py

#### Iterative sampling, connecting, and pruning with sst
![double_int_trees][double_int_trees]

#### Tracking paths to create a single loop
![double_int_LQR_tracking][double_int_LQR_tracking]

[double_int_trees]: https://github.com/luclarocque/kinodynamic-planning/blob/master/double_int_trees.gif "Iterative sampling of the state-space with connections decided by sst"
[double_int_LQR_tracking]: https://github.com/luclarocque/kinodynamic-planning/blob/master/double_int_LQR_tracking.gif "Tracking algorithm follows along disconnected paths forming a continuous loop"

Based on work by [Yanbo Li, Zakary Littlefield, and Kostas E. Bekris](https://journals.sagepub.com/doi/10.1177/0278364915614386).

