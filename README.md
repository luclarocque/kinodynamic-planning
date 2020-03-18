# Kinodynamic Planning

Master's research: motion planning implementations for quadrotors (drones) as well as other robots with nonlinear dynamics.

## fmt (kino-FMT*)
The kino-FMT* algorithm presented here is my implementation of the asymptotically optimal and probabilistically complete **real-time** kinodynamic planning algorithm used specifically for quadrotors.

Based on work by [Ross Allen and Marco Pavone](https://stanfordasl.github.io/wp-content/papercite-data/pdf/Allen.Pavone.AIAAGNC16.pdf).


## sst (SST*)
The SST* algorithm presented here is my implementation of the asymptotically optimal and probabilistically complete (but not real-time) kinodynamic planning algorithm for arbitrary nonlinear systems, i.e., any robot! The real key behind this algorithm is its ability to use only forward integration to solve motion planning problems for systems with nonlinear dynamics. This is unlike many other solvers which require solutions to difficult/impossible boundary value problems.

Based on work by [Yanbo Li, Zakary Littlefield, and Kostas E. Bekris](https://journals.sagepub.com/doi/10.1177/0278364915614386).

