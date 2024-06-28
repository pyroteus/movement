import warnings

import firedrake
import ufl
from firedrake.__future__ import interpolate

__all__ = ["MeshTanglingChecker"]


class MeshTanglingChecker:
    """
    A class for tracking whether a mesh has tangled, i.e. whether any of its elements
    have become inverted.
    """

    def __init__(self, mesh, raise_error=True):
        """
        :arg mesh: the mesh to track if tangled
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :kwarg raise_error: if ``True``, an error is raised if any element is tangled,
            otherwise a warning is raised
        :type raise_error: :class:`bool`
        """
        if mesh.topological_dimension() != 2:
            raise NotImplementedError("Tangling check only currently supported in 2D.")
        self.mesh = mesh
        self.raise_error = raise_error

        # Store initial signs of Jacobian determinant
        P0 = firedrake.FunctionSpace(mesh, "DG", 0)
        detJ = ufl.JacobianDeterminant(mesh)
        s = firedrake.assemble(interpolate(ufl.sign(detJ), P0))

        # Get scaled Jacobian expression
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        J = firedrake.assemble(interpolate(ufl.Jacobian(mesh), P0_ten))
        edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
        edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
        edge3 = edge1 - edge2
        norm1 = ufl.sqrt(ufl.dot(edge1, edge1))
        norm2 = ufl.sqrt(ufl.dot(edge2, edge2))
        norm3 = ufl.sqrt(ufl.dot(edge3, edge3))
        prod1 = ufl.max_value(norm1 * norm2, norm1 * norm3)
        prod2 = ufl.max_value(norm2 * norm3, norm2 * norm1)
        prod3 = ufl.max_value(norm3 * norm1, norm3 * norm2)
        maxval = ufl.max_value(ufl.max_value(prod1, prod2), prod3)
        self._sj_expr = detJ / maxval * s
        self._sj = firedrake.Function(P0)

    @property
    def scaled_jacobian(self):
        self._sj.interpolate(self._sj_expr)
        return self._sj

    def check(self):
        """
        Check whether any element orientations have changed since the tangling checker
        was created.
        """
        sj = self.scaled_jacobian.dat.data_with_halos
        num_tangled = len(sj[sj < 0])
        if num_tangled > 0:
            plural = "s" if num_tangled > 1 else ""
            msg = f"Mesh has {num_tangled} tangled element{plural}."
            if self.raise_error:
                raise ValueError(msg)
            warnings.warn(msg)
        return num_tangled
