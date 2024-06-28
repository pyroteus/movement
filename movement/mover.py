from warnings import warn

import firedrake
import firedrake.exceptions as fexc
import numpy as np
from firedrake.cython.dmcommon import create_section
from firedrake.petsc import PETSc

from movement.tangling import MeshTanglingChecker

__all__ = ["PrimeMover"]


class PrimeMover:
    """
    Base class for all mesh movers.
    """

    def __init__(
        self,
        mesh,
        monitor_function=None,
        raise_convergence_errors=True,
        tangling_check=None,
        **kwargs,
    ):
        r"""
        :arg mesh: the physical mesh
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg monitor_function: a Python function which takes a mesh as input
        :type monitor_function: :class:`~.Callable`
        :kwarg raise_convergence_errors: convergence error handling behaviour: if `True`
            then :class:`~.ConvergenceError`\s are raised, else warnings are raised and
            the program is allowed to continue
        :type raise_convergence_errors: :class:`bool`
        :kwarg tangling_check: check whether the mesh has tangled elements (by default
            on in the 2D case and off otherwise)
        :type tangling_check: :class:`bool`
        """
        self.mesh = firedrake.Mesh(mesh.coordinates.copy(deepcopy=True))
        self.monitor_function = monitor_function
        if not raise_convergence_errors:
            warn(
                f"{type(self)}.move called with raise_convergence_errors=False."
                " Beware: this option can produce poor quality meshes!"
            )
        self.raise_convergence_errors = raise_convergence_errors
        self.dim = self.mesh.topological_dimension()
        self.gdim = self.mesh.geometric_dimension()
        self.plex = self.mesh.topology_dm
        self.vertex_indices = self.plex.getDepthStratum(0)
        self.edge_indices = self.plex.getDepthStratum(1)

        # Measures
        degree = kwargs.get("quadrature_degree")
        self.dx = firedrake.dx(domain=self.mesh, degree=degree)
        self.ds = firedrake.ds(domain=self.mesh, degree=degree)
        self.dS = firedrake.dS(domain=self.mesh, degree=degree)

        # Mesh coordinate functions
        self.coord_space = self.mesh.coordinates.function_space()
        self.x = firedrake.Function(self.mesh.coordinates, name="Physical coordinates")
        self.xi = firedrake.Function(
            self.mesh.coordinates, name="Computational coordinates"
        )
        self.v = firedrake.Function(self.coord_space, name="Mesh velocity")

        # Utilities
        if tangling_check is None:
            tangling_check = self.dim == 2
        if tangling_check:
            self.tangling_checker = MeshTanglingChecker(
                self.mesh, raise_error=raise_convergence_errors
            )

    def _convergence_message(self, iterations=None):
        """
        Report solver convergence.

        :kwarg iterations: number of iterations before reaching convergence
        :type iterations: :class:`int`
        """
        msg = "Solver converged"
        if iterations:
            msg += f" in {iterations} iteration{plural(iterations)}"
        PETSc.Sys.Print(f"{msg}.")

    def _exception(self, msg, exception=None, error_type=fexc.ConvergenceError):
        """
        Raise an error or warning as indicated by the :attr:`raise_convergence_error`
        option.

        :arg msg: message for the error/warning report
        :type msg: :class:`str`
        :kwarg exception: original exception that was triggered
        :type exception: :class:`~.Exception` object
        :kwarg error_type: error class to use
        :type error_type: :class:`~.Exception` class
        """
        exc_type = error_type if self.raise_convergence_errors else Warning
        if exception:
            raise exc_type(msg) from exception
        else:
            raise exc_type(msg)

    def _convergence_error(self, iterations=None, exception=None):
        """
        Raise an error or warning for a solver fail as indicated by the
        :attr:`raise_convergence_error` option.

        :kwarg iterations: number of iterations before failure
        :type iterations: :class:`int`
        :kwarg exception: original exception that was triggered
        :type exception: :class:`~.Exception`
        """
        msg = "Solver failed to converge"
        if iterations:
            msg += f" in {iterations} iteration{plural(iterations)}"
        self._exception(f"{msg}.", exception=exception)

    def _divergence_error(self, iterations=None, exception=None):
        """
        Raise an error or warning for a solver divergence as indicated by the
        :attr:`raise_convergence_error` option.

        :kwarg iterations: number of iterations before failure
        :type iterations: :class:`int`
        :kwarg exception: original exception that was triggered
        :type exception: :class:`~.Exception`
        """
        msg = "Solver diverged"
        if iterations:
            msg += f" after {iterations} iteration{plural(iterations)}"
        self._exception(f"{msg}.", exception=exception)

    def _get_coordinate_section(self):
        entity_dofs = np.zeros(self.dim + 1, dtype=np.int32)
        entity_dofs[0] = self.gdim
        self._coordinate_section = create_section(self.mesh, entity_dofs)[0]
        dm_coords = self.plex.getCoordinateDM()
        dm_coords.setDefaultSection(self._coordinate_section)
        self._coords_local_vec = dm_coords.createLocalVec()
        self._update_plex_coordinates()

    def _update_plex_coordinates(self):
        if not hasattr(self, "_coords_local_vec"):
            self._get_coordinate_section()
        self._coords_local_vec.array[:] = np.reshape(
            self.mesh.coordinates.dat.data_with_halos,
            self._coords_local_vec.array.shape,
        )
        self.plex.setCoordinatesLocal(self._coords_local_vec)

    def _get_edge_vector_section(self):
        entity_dofs = np.zeros(self.dim + 1, dtype=np.int32)
        entity_dofs[1] = 1
        self._edge_vector_section = create_section(self.mesh, entity_dofs)[0]

    def coordinate_offset(self, index):
        """
        Get the DMPlex coordinate section offset
        for a given `index`.
        """
        if not hasattr(self, "_coordinate_section"):
            self._get_coordinate_section()
        return self._coordinate_section.getOffset(index) // self.dim

    def edge_vector_offset(self, index):
        """
        Get the DMPlex edge vector section offset
        for a given `index`.
        """
        if not hasattr(self, "_edge_vector_section"):
            self._get_edge_vector_section()
        return self._edge_vector_section.getOffset(index)

    def coordinate(self, index):
        """
        Get the mesh coordinate associated with
        a given `index`.
        """
        return self.mesh.coordinates.dat.data_with_halos[self.get_offset(index)]

    def move(self):
        """
        Move the mesh according to the method of choice.
        """
        raise NotImplementedError("Implement `move` in the derived class.")

    def adapt(self):
        """
        Alias of `move`.
        """
        warn(
            "`adapt` is deprecated (use `move` instead)",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.move()


def plural(iterations):
    return "s" if iterations != 1 else ""
