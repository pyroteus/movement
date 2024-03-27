import firedrake
from firedrake.cython.dmcommon import create_section
import numpy as np


__all__ = ["PrimeMover"]


class PrimeMover(object):
    """
    Base class for all mesh movers.
    """

    def __init__(self, mesh, monitor_function=None, **kwargs):
        self.mesh = firedrake.Mesh(mesh.coordinates.copy(deepcopy=True))
        self.monitor_function = monitor_function
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
        self._x = firedrake.Function(self.mesh.coordinates, name="Physical coordinates")
        self.xi = firedrake.Function(
            self.mesh.coordinates, name="Computational coordinates"
        )
        self.v = firedrake.Function(self.coord_space, name="Mesh velocity")

    def _get_coordinate_section(self):
        entity_dofs = np.zeros(self.dim + 1, dtype=np.int32)
        entity_dofs[0] = self.gdim
        self._coordinate_section = create_section(self.mesh, entity_dofs)
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
        self._edge_vector_section = create_section(self.mesh, entity_dofs)

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
        from warnings import warn

        warn(
            "`adapt` is deprecated (use `move` instead)",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.move()
