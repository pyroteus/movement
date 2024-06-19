import unittest

from firedrake import *

from movement import *


class TestTangling(unittest.TestCase):
    """
    Unit tests for mesh tangling checking.
    """

    def test_dim_notimplementederror(self):
        with self.assertRaises(NotImplementedError) as cm:
            MeshTanglingChecker(UnitIntervalMesh(1))
        msg = "Tangling check only currently supported in 2D."
        self.assertEqual(str(cm.exception), msg)

    def test_tangling_checker_error1(self):
        mesh = UnitSquareMesh(3, 3)
        checker = MeshTanglingChecker(mesh, raise_error=True)
        mesh.coordinates.dat.data[3] += 0.2
        with self.assertRaises(ValueError) as cm:
            checker.check()
        msg = "Mesh has 1 tangled element."
        self.assertEqual(str(cm.exception), msg)

    def test_tangling_checker_error2(self):
        mesh = UnitSquareMesh(3, 3)
        checker = MeshTanglingChecker(mesh, raise_error=True)
        mesh.coordinates.dat.data[3] += 0.5
        with self.assertRaises(ValueError) as cm:
            checker.check()
        msg = "Mesh has 3 tangled elements."
        self.assertEqual(str(cm.exception), msg)

    def test_tangling_checker_warning1(self):
        mesh = UnitSquareMesh(3, 3)
        checker = MeshTanglingChecker(mesh, raise_error=False)
        self.assertEqual(checker.check(), 0)
        mesh.coordinates.dat.data[3] += 0.2
        self.assertEqual(checker.check(), 1)
