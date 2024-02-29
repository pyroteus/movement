import firedrake
from firedrake.petsc import PETSc
import ufl
import movement.solver_parameters as solver_parameters
from movement.mover import PrimeMover
import numpy as np


__all__ = ["LaplacianSmoother"]


class LaplacianSmoother(PrimeMover):
    """
    Movement of a ``mesh`` is driven by a mesh velocity, which is determined by solving
    a Poisson problem.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, timestep, **kwargs):
        super().__init__(mesh, **kwargs)
        assert timestep > 0.0
        self.dt = timestep
        self.f = firedrake.Function(self.coord_space)
        dim = self.mesh.topological_dimension()
        self.displacement = np.zeros((self.mesh.num_vertices(), dim))

    @PETSc.Log.EventDecorator()
    def _setup_solver(self, fixed_boundaries=[]):
        if not hasattr(self, "_solver"):
            test = firedrake.TestFunction(self.coord_space)
            trial = firedrake.TrialFunction(self.coord_space)
            a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * self.dx
            L = ufl.inner(self.f, test) * self.dx
            bcs = firedrake.DirichletBC(self.coord_space, 0, fixed_boundaries)
            problem = firedrake.LinearVariationalProblem(a, L, self.v, bcs=bcs)
            self._solver = firedrake.LinearVariationalSolver(
                problem,
                solver_parameters=solver_parameters.cg,
            )
        self._solver.solve()

    @PETSc.Log.EventDecorator()
    def move(self, time, update_forcings=None, fixed_boundaries=[]):
        """
        Assemble and solve the Laplacian system and update the coordinates.

        :arg time: the current time
        :kwarg update_forcings: function that updates the forcing :attr:`f` at the
            current time
        :kwarg fixed_boundaries: list of boundaries where Dirichlet conditions are to be
            enforced
        """
        if update_forcings is not None:
            update_forcings(time)
        self._setup_solver(fixed_boundaries=fixed_boundaries)

        # Solve on computational mesh
        self.mesh.coordinates.assign(self.xi)
        self._solver.solve()

        # Update mesh coordinates
        self.displacement[:] = self.v.dat.data_with_halos * self.dt
        self._x.dat.data_with_halos[:] += self.displacement
        self.mesh.coordinates.assign(self._x)
