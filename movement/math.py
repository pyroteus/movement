import numpy as np
import sympy

__all__ = []


def equation_of_line(a, b):
    """
    Deduce an expression for the equation of a line passing through two points.

    :arg a: the first point the line passes through
    :type a: :class:`tuple`
    :arg b: the second point the line passes through
    :type b: :class:`tuple`
    :returns: a function of two variables representing the line
    :rtype: :class:`~.Callable`
    """
    try:
        line = sympy.Line(sympy.Point2D(a), sympy.Point2D(b))
    except ValueError as exc:
        raise ValueError("Could not determine a line for the provided points.") from exc

    def equation(x, y):
        return line.distance(sympy.Point2D((x, y)))

    return equation


def _equation_of_plane(a, b, c):
    """
    Deduce an expression for the equation of a plane passing through three points.

    Returns `None` if the points are colinear.

    :arg a: the first point the line passes through
    :type a: :class:`tuple`
    :arg b: the second point the line passes through
    :type b: :class:`tuple`
    :arg c: the third point the line passes through
    :type c: :class:`tuple`
    :returns: a function of three variables representing the plane
    :rtype: :class:`~.Callable`
    """
    plane = sympy.Plane(sympy.Point3D(a), sympy.Point3D(b), sympy.Point3D(c))

    def equation(x, y, z):
        return plane.distance(sympy.Point3D((x, y, z)))

    return equation


def equation_of_plane(*points):
    r"""
    Deduce an expression for the equation of a plane passing through a set of points.

    :arg points: the first point the line passes through
    :type points: :class:`tuple` of :class:`tuple`\s
    :returns: a function of three variables representing the plane
    :rtype: :class:`~.Callable`
    """
    assert len(points) >= 3
    indices = list(range(len(points)))
    while len(indices) >= 3:
        np.random.shuffle(indices)
        i, j, k = indices[:3]
        try:
            return _equation_of_plane(points[i], points[j], points[k])
        except ValueError:
            indices.pop(0)
    raise ValueError("Could not determine a plane for the provided points.")
