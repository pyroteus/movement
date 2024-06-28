import firedrake
import firedrake.exceptions as fexc
import numpy as np
import ufl
from firedrake.petsc import PETSc

import movement.solver_parameters as solver_parameters
from movement.mover import PrimeMover

__all__ = ["LaplacianSmoother"]


class LaplacianSmoother(PrimeMover):
    r"""
    Movement of a ``mesh`` is driven by a mesh velocity :math:`\mathbf{v}`, which is
    determined by solving  a vector Laplace problem

    .. math::
        \nabla^2_{\boldsymbol{\xi}}\mathbf{v} = \boldsymbol{0}, \quad \boldsymbol{\xi}\in\Omega,

    under non-zero Dirichlet boundary conditions on a forced boundary section
    :math:`\partial\Omega_f` and zero Dirichlet boundary conditions elsewhere:

    .. math::
        \mathbf{v} = \left\{\begin{array}{rl}
            \mathbf{v}_D, & \boldsymbol{\xi}\in\partial\Omega_f\\
            \boldsymbol{0}, & \boldsymbol{\xi}\in\partial\Omega\backslash\partial\Omega_f
        \end{array}\right.

    where the computational coordinates :math:`\boldsymbol{\xi} := \mathbf{x}(t_0)` are
    the physical coordinates at the beginning of the simulation.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, timestep, **kwargs):
        """
        :arg mesh: the physical mesh to be moved
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg timestep: the timestep length used
        :type timestep: :class:`float`
        """
        super().__init__(mesh, **kwargs)
        assert timestep > 0.0
        self.dt = timestep
        dim = self.mesh.topological_dimension()
        self.displacement = np.zeros((self.mesh.num_vertices(), dim))

    @PETSc.Log.EventDecorator()
    def _setup_solver(self, boundary_conditions):
        if not hasattr(self, "_solver"):
            test = firedrake.TestFunction(self.coord_space)
            trial = firedrake.TrialFunction(self.coord_space)
            f = firedrake.Function(self.coord_space, name="Zero RHS")

            a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * self.dx
            L = ufl.inner(f, test) * self.dx
            problem = firedrake.LinearVariationalProblem(
                a, L, self.v, bcs=boundary_conditions
            )
            self._solver = firedrake.LinearVariationalSolver(
                problem,
                solver_parameters=solver_parameters.cg_ilu,
            )
        self._solver.solve()

    @PETSc.Log.EventDecorator()
    def move(self, time, update_boundary_velocity=None, boundary_conditions=None):
        """
        Assemble and solve the Laplacian system and update the coordinates.

        :arg time: the current time
        :type time: :class:`float`
        :kwarg update_boundary_velocity: function that updates the boundary conditions at
            the current time
        :type update_boundary_velocity: :class:`~.Callable` with a single argument of
            :class:`float` type
        :kwarg boundary_conditions: Dirichlet boundary conditions to be enforced
        :type boundary_conditions: :class:`~.DirichletBC` or :class:`list` thereof
        """
        if update_boundary_velocity is not None:
            update_boundary_velocity(time)
        if not boundary_conditions:
            boundary_conditions = firedrake.DirichletBC(
                self.coord_space, 0, "on_boundary"
            )
        self._setup_solver(boundary_conditions)

        # Solve on computational mesh
        self.mesh.coordinates.assign(self.xi)
        try:
            self._solver.solve()
        except fexc.ConvergenceError as conv_err:
            self._convergence_error(exception=conv_err)

        # Update mesh coordinates
        self.displacement[:] = self.v.dat.data_with_halos * self.dt
        self.x.dat.data_with_halos[:] += self.displacement
        self.mesh.coordinates.assign(self.x)

        if hasattr(self, "tangling_checker"):
            self.tangling_checker.check()
