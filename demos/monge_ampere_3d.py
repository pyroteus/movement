# Monge-Ampère mesh movement in three dimensions
# ====================================================================

# In this demo we demonstrate that the  Monge-Ampère mesh movement
# can also be applied to 3D meshes. We employ the `sinatan3` function
# from :cite:`park2019` to introduce an interesting pattern.

import movement
from firedrake import *


def sinatan3(mesh):
    x, y, z = SpatialCoordinate(mesh)
    return 0.1*sin(50*x*z)+atan2(0.1, sin(5*y)-2*x*z)


# We will now try to use mesh movement to optimize the mesh such that it can
# most accurately represent this function with limited resolution.
# A good indicator for where resolution is required
# is to look at the curvature of the function which can be expressed
# in terms of the norm of the Hessian. A monitor function
# that targets high resolution in places of high curvature then looks like
#
# .. math::
#
#    m = 1 + \alpha \frac{H(u_h):H(u_h)}{\max_{{\bf x}\in\Omega} H(u_h):H(u_h)}
#
# where :math:`:` indicates the inner product, i.e. :math:`\sqrt{H:H}` is the Frobenius norm
# of :math:`H`. We have normalized such that the minimum of the monitor function is one (where
# the error is zero), and its maximum is :math:`1 + \alpha` (where the curvature is maximal). This
# means we can select the ratio between the largest and smallest cell volume  in the
# moved mesh as :math:`1+\alpha`.
#
# As in the `previous Monge-Ampère demo <./monge_ampere1.py.html>`__, we use the
# MongeAmpereMover to perform the mesh movement based on this monitor. We need
# to provide the monitor as a callback function that takes the mesh as its
# input. During the iterations of the mesh movement process the monitor will then
# be re-evaluated in the (iteratively)
# moved mesh nodes so that, as we improve the mesh, we can also more accurately
# express the monitor function in the desired high-resolution areas.
# To track what happens during these iterations, we define a VTK file object
# that we will write to in every call when the monitor gets re-evaluated.

from firedrake.output import VTKFile
f = VTKFile('monitor.pvd')
alpha = Constant(10.)


def monitor(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    # interpolation of the function itself, for output purposes only:
    u = Function(V, name='sinatan3')
    u.interpolate(sinatan3(mesh))

    # NOTE: we are taking the Hessian of the _symbolic_ UFL expression
    # returned by sinatan3(mesh), *not* of the P1 interpolated version
    # stored in u. u is a piecewise linear approximation; Taking its gradient
    # once would result in a piecewise constant (cell-wise) gradient, and taking
    # the gradient of that again would simply be zero.
    hess = grad(grad(sinatan3(mesh)))

    hnorm = Function(V, name='hessian_norm')
    hnorm.interpolate(inner(hess, hess))
    hmax = hnorm.dat.data[:].max()
    f.write(u, hnorm)
    return 1.0 + alpha * hnorm/Constant(hmax)


# Let us try this on a fairly coarse unit cube mesh. Note that
# we use the `"relaxation"` method (see :cite:`McRae:20181),
# which gives faster convergence for this case.

n = 20
mesh = UnitCubeMesh(n, n, n)
mover = movement.MongeAmpereMover(mesh, monitor, method="relaxation")
mover.move()

# The results will be written to the `monitor.pvd` file which represents
# a series of outputs storing the function and hessian norm evaluated
# on each of the iteratively moved meshes. This can be viewed, e.g., using
# [ParaView](https://www.paraview.org/) to produce the following
# image:
#
# .. figure:: monge_ampere_3d-paraview.jpg
#    :figwidth: 60%
#    :align: center
#
# This tutorial can be dowloaded as a `Python script <monge_ampere_3d.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography:: demo_references.bib
#    :filter: docname in docnames
