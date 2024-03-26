import movement
from firedrake import *

mesh = UnitCubeMesh(10, 10, 10)
def sinfun3_expr(mesh):
    x, y, z = SpatialCoordinate(mesh)
    xyz = (x-0.4)*(y-0.4)*(z-0.4)
    sinfun3 = conditional(xyz < pi/50, 0.1 * sin(50*xyz),
                          conditional(xyz< 2*pi/50, sin(50*xyz),
                                      0.1*sin(50*xyz)))
    return sinfun3

V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name='sinfun3')
u.interpolate(sinfun3_expr(mesh))

f = File('tmp.pvd')
def monitor(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='sinfun3')
    hess = grad(grad(sinfun3_expr(mesh)))
    u.interpolate(inner(hess, hess))
    umax = u.dat.data[:].max()
    f.write(u)
    return 1.0 + 5 * u/umax


mover = movement.MongeAmpereMover(mesh, monitor, method="quasi_newton")
mover.move()

