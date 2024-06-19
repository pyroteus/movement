import unittest

from movement.math import equation_of_line, equation_of_plane


class TestLine(unittest.TestCase):
    """
    Unit tests for :func:`~.equation_of_line`.
    """

    def _loop_over_grid(self, f, condition):
        for i in range(10):
            for j in range(10):
                if condition(i, j):
                    self.assertAlmostEqual(f(i, j), 0)
                else:
                    self.assertNotEqual(f(i, j), 0)

    def test_duplicate_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            equation_of_line((0, 0), (0, 0))
        msg = "Could not determine a line for the provided points."
        self.assertEqual(str(cm.exception), msg)

    def test_x_equals_y(self):
        f = equation_of_line((0, 0), (1, 1))

        def condition(x, y):
            return x == y

        self._loop_over_grid(f, condition)

    def test_x_equals_zero(self):
        f = equation_of_line((0, 0), (0, 1))

        def condition(x, y):
            return x == 0

        self._loop_over_grid(f, condition)

    def test_y_equals_zero(self):
        f = equation_of_line((0, 0), (1, 0))

        def condition(x, y):
            return y == 0

        self._loop_over_grid(f, condition)


class TestPlane(unittest.TestCase):
    """
    Unit tests for :func:`~.equation_of_plane`.
    """

    def _loop_over_grid(self, f, condition):
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    if condition(i, j, k):
                        self.assertAlmostEqual(f(i, j, k), 0)
                    else:
                        self.assertNotEqual(f(i, j, k), 0)

    def test_2duplicates_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            equation_of_plane((0, 0, 0), (1, 1, 1), (1, 1, 1))
        msg = "Could not determine a plane for the provided points."
        self.assertEqual(str(cm.exception), msg)

    def test_3duplicates_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            equation_of_plane((0, 0, 0), (0, 0, 0), (0, 0, 0))
        msg = "Could not determine a plane for the provided points."
        self.assertEqual(str(cm.exception), msg)

    def test_colinear_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            equation_of_plane((0, 0, 0), (1, 1, 1), (2, 2, 2))
        msg = "Could not determine a plane for the provided points."
        self.assertEqual(str(cm.exception), msg)

    def test_unit_triangle_xy(self):
        f = equation_of_plane((0, 0, 0), (1, 0, 0), (0, 1, 0))

        def condition(x, y, z):
            return z == 0

        self._loop_over_grid(f, condition)

    def test_unit_triangle_yz(self):
        f = equation_of_plane((0, 0, 0), (0, 1, 0), (0, 0, 1))

        def condition(x, y, z):
            return x == 0

        self._loop_over_grid(f, condition)

    def test_unit_triangle_zx(self):
        f = equation_of_plane((0, 0, 0), (0, 0, 1), (1, 0, 0))

        def condition(x, y, z):
            return y == 0

        self._loop_over_grid(f, condition)

    def test_x_plus_y_plus_z_equals_zero(self):
        f = equation_of_plane((0, 0, 0), (1, -1, 0), (0, 1, -1))

        def condition(x, y, z):
            return x + y + z == 0

        self._loop_over_grid(f, condition)
