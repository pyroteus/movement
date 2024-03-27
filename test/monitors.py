from firedrake import *


def const_monitor(mesh):
    return Constant(1.0)


def ring_monitor(mesh):
    alpha = Constant(10.0)  # amplitude
    beta = Constant(200.0)  # width
    gamma = Constant(0.15)  # radius
    dim = mesh.geometric_dimension()
    xyz = SpatialCoordinate(mesh) - as_vector([0.5]*dim)
    r = dot(xyz, xyz)
    return Constant(1.0) + alpha / cosh(beta * (r - gamma)) ** 2
