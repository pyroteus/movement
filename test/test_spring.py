import unittest

import numpy as np
from firedrake import *

from movement import SpringMover


class TestSetup(unittest.TestCase):
    """
    Unit tests for setting up spring-based Movers.
    """

    def setUp(self):
        self.mesh = UnitTriangleMesh()

    def test_method_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            SpringMover(self.mesh, 1.0, method="method")
        self.assertEqual(str(cm.exception), "Method 'method' not recognised.")

    def test_torsional_notimplementederror(self):  # TODO: (#36)
        with self.assertRaises(NotImplementedError) as cm:
            SpringMover(self.mesh, 1.0, method="torsional")
        self.assertEqual(str(cm.exception), "Torsional springs not yet implemented.")

    def test_invalid_bc_space_valueerror(self):
        mover = SpringMover(self.mesh, 1.0)
        bc = DirichletBC(VectorFunctionSpace(self.mesh, "CG", 2), 0, "on_boundary")
        with self.assertRaises(ValueError) as cm:
            mover.assemble_stiffness_matrix(bc)
        msg = (
            "Boundary conditions must have SpringMover_Lineal.coord_space as their"
            " function space."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_invalid_bc_tag_valueerror(self):
        mover = SpringMover(self.mesh, 1.0)
        bc = DirichletBC(mover.coord_space, 0, 4)
        with self.assertRaises(ValueError) as cm:
            mover.assemble_stiffness_matrix(bc)
        msg = "[4] contains invalid boundary tags."
        self.assertEqual(str(cm.exception), msg)


class TestStiffness(unittest.TestCase):
    """
    Unit tests for the stiffness matrices used by the spring-based Movers.
    """

    def test_boundary_conditions_triangle_all_boundary(self):
        mesh = UnitTriangleMesh()
        mover = SpringMover(mesh, 1.0, method="lineal")
        self.assertTrue(np.allclose(mover._stiffness_matrix(), 2 * np.eye(6)))
        self.assertTrue(np.allclose(mover.assemble_stiffness_matrix(), np.eye(6)))

    def test_boundary_conditions_triangle_one_segment(self):
        mesh = UnitTriangleMesh()
        mover = SpringMover(mesh, 1.0, method="lineal")
        self.assertTrue(np.allclose(mover._stiffness_matrix(), 2 * np.eye(6)))
        bc = DirichletBC(mover.coord_space, 0, 1)
        expected = np.diag([2, 2, 1, 1, 1, 1])
        self.assertTrue(np.allclose(mover.assemble_stiffness_matrix(bc), expected))


class TestQuantities(unittest.TestCase):
    """
    Unit tests for the element quantity utilities underlying the :class:`~.SpringMover`
    classes.
    """

    def setUp(self):
        self.mover = SpringMover(UnitTriangleMesh(), 1.0)

    @staticmethod
    def rad2deg(radians):
        return 180 * radians / np.pi

    def test_facet_areas(self):
        """
        Test that the edge lenths of a right-angled triangle are computed correctly.
        """
        facet_areas = self.mover.facet_areas.dat.data
        self.assertTrue(np.allclose(facet_areas, [1, np.sqrt(2), 1]))

    def test_tangents(self):
        """
        Test that the tangent vectors of a right-angled triangle are computed correctly.
        """
        tangents = self.mover.tangents.dat.data
        expected = [[1, 0], [np.sqrt(2) / 2, np.sqrt(2) / 2], [0, 1]]
        self.assertTrue(np.allclose(np.abs(tangents), expected))

    def test_angles(self):
        """
        Test that the arguments (angles from the x-axis) of a right-angled triangle are
        computed correctly.
        """
        angles = self.rad2deg(self.mover.angles.dat.data)
        self.assertTrue(np.allclose(angles, [0, 135, 90]))
