import firedrake
import firedrake.function as ffunc
from firedrake.petsc import PETSc
import ufl
from collections.abc import Iterable
import numpy as np
import movement.solver_parameters as solver_parameters
from movement.mover import PrimeMover


__all__ = ["SpringMover_Lineal", "SpringMover_Torsional", "SpringMover"]


def SpringMover(mesh, method="lineal", **kwargs):
    """
    Movement of a ``mesh`` is determined by reinterpreting it as a structure of stiff
    beams and solving an associated discrete linear elasticity problem.

    See Farhat, Degand, Koobus and Lesoinne, "Torsional springs for two-dimensional
    dynamic unstructured fluid meshes" (1998), Computer methods in applied mechanics and
    engineering, 163:231-245.

    :arg mesh: the mesh to be moved
    :kwarg method: flavour of spring-based method to use
    """
    if method == "lineal":
        return SpringMover_Lineal(mesh, **kwargs)
    elif method == "torsional":
        return SpringMover_Torsional(mesh, **kwargs)
    else:
        raise ValueError(f"Method {method} not recognised.")


class SpringMover_Base(PrimeMover):
    """
    Base class for mesh movers based on spring analogies.
    """

    def __init__(self, mesh, **kwargs):
        """
        :arg mesh: the physical mesh
        """
        super().__init__(mesh)
        self.HDivTrace = firedrake.FunctionSpace(self.mesh, "HDiv Trace", 0)
        self.HDivTrace_vec = firedrake.VectorFunctionSpace(self.mesh, "HDiv Trace", 0)
        self.f = ffunc.Function(self.mesh.coordinates.function_space())
        self.displacement = np.zeros(self.mesh.num_vertices())

    @property
    @PETSc.Log.EventDecorator()
    def facet_areas(self):
        """
        Compute the areas of all facets in the mesh.

        In 2D, this corresponds to edge lengths.
        """
        if not hasattr(self, "_facet_area_solver"):
            test = firedrake.TestFunction(self.HDivTrace)
            trial = firedrake.TrialFunction(self.HDivTrace)
            self._facet_area = ffunc.Function(self.HDivTrace)
            A = ufl.FacetArea(self.mesh)
            a = trial("+") * test("+") * self.dS + trial * test * self.ds
            L = test("+") * A * self.dS + test * A * self.ds
            prob = firedrake.LinearVariationalProblem(a, L, self._facet_area)
            self._facet_area_solver = firedrake.LinearVariationalSolver(
                prob,
                solver_parameters=solver_parameters.jacobi,
            )
        self._facet_area_solver.solve()
        return self._facet_area

    @property
    @PETSc.Log.EventDecorator()
    def tangents(self):
        """
        Compute tangent vectors for all edges in the mesh.
        """
        if not hasattr(self, "_tangents_solver"):
            test = firedrake.TestFunction(self.HDivTrace_vec)
            trial = firedrake.TrialFunction(self.HDivTrace_vec)
            self._tangents = ffunc.Function(self.HDivTrace_vec)
            n = ufl.FacetNormal(self.mesh)
            s = ufl.perp(n)
            a = (
                ufl.inner(trial("+"), test("+")) * self.dS
                + ufl.inner(trial, test) * self.ds
            )
            L = ufl.inner(test("+"), s("+")) * self.dS + ufl.inner(test, s) * self.ds
            prob = firedrake.LinearVariationalProblem(a, L, self._tangents)
            self._tangents_solver = firedrake.LinearVariationalSolver(
                prob,
                solver_parameters=solver_parameters.jacobi,
            )
        self._tangents_solver.solve()
        return self._tangents

    @property
    @PETSc.Log.EventDecorator()
    def angles(self):
        r"""
        Compute the argument of each edge in the mesh, i.e. its angle from the
        :math:`x`-axis in the :math:`x-y` plane.
        """
        t = self.tangents
        if not hasattr(self, "_angles_solver"):
            test = firedrake.TestFunction(self.HDivTrace)
            trial = firedrake.TrialFunction(self.HDivTrace)
            self._angles = ffunc.Function(self.HDivTrace)
            e0 = np.zeros(self.dim)
            e0[0] = 1.0
            X = ufl.as_vector(e0)
            a = trial("+") * test("+") * self.dS + trial * test * self.ds
            L = (
                test("+") * ufl.dot(t("+"), X("+")) * self.dS
                + test * ufl.dot(t, X) * self.ds
            )
            prob = firedrake.LinearVariationalProblem(a, L, self._angles)
            self._angles_solver = firedrake.LinearVariationalSolver(
                prob,
                solver_parameters=solver_parameters.jacobi,
            )
        self._angles_solver.solve()
        ones = np.ones_like(self._angles.dat.data)
        self._angles.dat.data[:] = np.maximum(
            np.minimum(self._angles.dat.data, ones), -ones
        )
        self._angles.dat.data[:] = np.arccos(self._angles.dat.data)
        return self._angles

    @property
    @PETSc.Log.EventDecorator()
    def stiffness_matrix(self):
        angles = self.angles
        edge_lengths = self.facet_areas
        bnd = self.mesh.exterior_facets
        N = self.mesh.num_vertices()

        K = np.zeros((2 * N, 2 * N))
        for e in range(*self.edge_indices):
            off = self.edge_vector_offset(e)
            i, j = (self.coordinate_offset(v) for v in self.plex.getCone(e))
            if bnd.point2facetnumber[e] != -1:
                K[2 * i][2 * i] += 1.0
                K[2 * i + 1][2 * i + 1] += 1.0
                K[2 * j][2 * j] += 1.0
                K[2 * j + 1][2 * j + 1] += 1.0
            else:
                l = edge_lengths.dat.data_with_halos[off]
                angle = angles.dat.data_with_halos[off]
                c = np.cos(angle)
                s = np.sin(angle)
                K[2 * i][2 * i] += c * c / l
                K[2 * i][2 * i + 1] += s * c / l
                K[2 * i][2 * j] += -c * c / l
                K[2 * i][2 * j + 1] += -s * c / l
                K[2 * i + 1][2 * i] += s * c / l
                K[2 * i + 1][2 * i + 1] += s * s / l
                K[2 * i + 1][2 * j] += -s * c / l
                K[2 * i + 1][2 * j + 1] += -s * s / l
                K[2 * j][2 * i] += -c * c / l
                K[2 * j][2 * i + 1] += -s * c / l
                K[2 * j][2 * j] += c * c / l
                K[2 * j][2 * j + 1] += s * c / l
                K[2 * j + 1][2 * i] += -s * c / l
                K[2 * j + 1][2 * i + 1] += -s * s / l
                K[2 * j + 1][2 * j] += s * c / l
                K[2 * j + 1][2 * j + 1] += s * s / l
        return K

    @PETSc.Log.EventDecorator()
    def apply_dirichlet_conditions(self, boundary_conditions=None):
        """
        Enforce that nodes on certain tagged boundaries do not move.

        :kwarg boundary_conditions: Dirichlet boundary conditions to be enforced
        :type boundary_conditions: :class:`~.DirichletBC` or :class:`list` thereof
        """
        if not boundary_conditions:
            boundary_conditions = firedrake.DirichletBC(
                self.coord_space, 0, "on_boundary"
            )
        if isinstance(boundary_conditions, firedrake.DirichletBC):
            boundary_conditions = [boundary_conditions]
        assert isinstance(boundary_conditions, Iterable)

        # Loop over each boundary condition provided
        for boundary_condition in boundary_conditions:
            if boundary_condition.function_space() != self.coord_space:
                raise ValueError(
                    f"Boundary conditions must have {type(self)}.coord_space as their"
                    " function space"
                )

            # Determine boundary subsets for the associated tags
            tags = boundary_condition.sub_domain
            if not isinstance(tags, Iterable):
                tags = [tags]
            bnd = self.mesh.exterior_facets
            if not set(tags).issubset(set(bnd.unique_markers)):
                raise ValueError(f"{tags} contains invalid boundary tags")
            subsets = sum(
                [list(bnd.subset(physID).indices) for physID in tags], start=[]
            )

            # Get vertex-based boundary data to be enforced
            boundary_value = boundary_condition._original_arg
            if not isinstance(boundary_value, ffunc.Function):
                boundary_value = ffunc.Function(self.coord_space).assign(boundary_value)
            boundary_data = boundary_value.dat.data

            # Loop over boundary edges and enforce the boundary values at their vertices
            for e in range(*self.edge_indices):
                i, j = (self.coordinate_offset(v) for v in self.plex.getCone(e))
                if bnd.point2facetnumber[e] in subsets:
                    self.displacement[2 * i] = boundary_data[i, 0]
                    self.displacement[2 * i + 1] = boundary_data[i, 1]
                    self.displacement[2 * j] = boundary_data[j, 0]
                    self.displacement[2 * j + 1] = boundary_data[j, 1]


class SpringMover_Lineal(SpringMover_Base):
    """
    Movement of a ``mesh`` is determined by reinterpreting it as a structure of stiff
    beams and solving an associated discrete linear elasticity problem.

    We consider the 'lineal' case, as described in Farhat, Degand, Koobus and Lesoinne,
    "Torsional springs for two-dimensional dynamic unstructured fluid meshes" (1998),
    Computer methods in applied mechanics and engineering, 163:231-245.
    """

    @PETSc.Log.EventDecorator()
    def move(self, time, update_forcings=None, boundary_conditions=None):
        """
        Assemble and solve the lineal spring system and update the coordinates.

        :arg time: the current time
        :type time: :class:`float`
        :kwarg update_forcings: function that updates the forcing :attr:`f` and/or
            boundary conditions at the current time
        :type update_forcings: :class:`~.Callable` with a single argument of
            :class:`float` type
        :kwarg boundary_conditions: Dirichlet boundary conditions to be enforced
        :type boundary_conditions: :class:`~.DirichletBC` or :class:`list` thereof
        """
        if update_forcings is not None:
            update_forcings(time)

        # Assemble
        K = self.stiffness_matrix
        rhs = self.f.dat.data.flatten()

        # Solve
        self.displacement = np.linalg.solve(K, rhs)

        # Enforce Dirichlet conditions as a post-process
        self.apply_dirichlet_conditions(boundary_conditions)

        # Update mesh coordinates
        shape = self.mesh.coordinates.dat.data_with_halos.shape
        self.mesh.coordinates.dat.data_with_halos[:] += self.displacement.reshape(shape)
        self._update_plex_coordinates()


class SpringMover_Torsional(SpringMover_Lineal):
    """
    Movement of a ``mesh`` is determined by reinterpreting it as a structure of stiff
    beams and solving an associated discrete linear elasticity problem.

    We consider the 'torsional' case, as described in Farhat, Degand, Koobus and
    Lesoinne, "Torsional springs for two-dimensional dynamic unstructured fluid meshes"
    (1998), Computer methods in applied mechanics and engineering, 163:231-245.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Torsional springs not yet implemented")  # TODO
