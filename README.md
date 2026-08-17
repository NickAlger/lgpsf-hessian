# lgpsf-petsc
Laguerre-Gaussian point spread function Hessian approximation in distributed memory parallel using petsc backend. Includes conversion to distributed global low rank matrix and downstream linear algebra operations such as matrix solves, square root applies, and computation of the log determinant in the prior weighted space. The lgpsf approximations are done using the lgpsf library per compute node. Requires diagonal (lumped) mass matrix.

