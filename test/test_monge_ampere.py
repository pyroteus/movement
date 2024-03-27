from movement import *
from monitors import *
import pytest
import numpy as np


@pytest.fixture(params=["relaxation", "quasi_newton"])
def method(request):
    return request.param


@pytest.fixture(params=[True, False])
def fix_boundary(request):
    return request.param


@pytest.fixture(params=[2, 3])
def mesh_dim(request):
    if request.param == 2:
        def mesh(n):
            return UnitSquareMesh(n, n)
    else:
        def mesh(n):
            return UnitCubeMesh(n, n, n)
    return mesh


def test_uniform_monitor(mesh_dim, method, exports=False):
    """
    Test that the mesh mover converges in one
    iteration for a constant monitor function.
    """
    n = 10
    mesh = mesh_dim(n)
    coords = mesh.coordinates.dat.data.copy()

    mover = MongeAmpereMover(mesh, const_monitor, method=method)
    num_iterations = mover.move()

    assert np.allclose(coords, mover.mesh.coordinates.dat.data)
    assert num_iterations == 0


def test_continue(mesh_dim, method, exports=False):
    """
    Test that providing a good initial guess
    benefits the solver.
    """
    n = 20
    mesh = mesh_dim(n)
    rtol = 1.0e-03

    # Solve the problem to a weak tolerance
    mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=0.1)
    num_it_init = mover.move()
    if exports:
        File("outputs/init.pvd").write(mover.phi, mover.sigma)

    # Continue with a tighter tolerance
    mover.rtol = rtol
    num_it_continue = mover.move()
    if exports:
        File("outputs/continue.pvd").write(mover.phi, mover.sigma)

    # Solve the problem again to a tight tolerance
    mesh = mesh_dim(n)
    mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=rtol)
    num_it_naive = mover.move()
    if exports:
        File("outputs/naive.pvd").write(mover.phi, mover.sigma)

    assert num_it_continue <= num_it_naive
    assert num_it_init + num_it_continue <= num_it_naive
    # FIXME: Looks like the mesh is tangled or close to tangling
    #        for the relaxation method, which is concerning.


def test_change_monitor(mesh_dim, method, exports=True):
    """
    Test that the mover can handle changes to
    the monitor function, such as would happen
    during timestepping.
    """
    n = 20
    mesh = mesh_dim(n)
    dim = mesh.geometric_dimension()
    coords = mesh.coordinates.dat.data.copy()
    tol = 1.0e-03

    # Adapt to a ring monitor
    mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=1e3*tol**dim)
    mover.move()
    if exports:
        File("outputs/ring.pvd").write(mover.phi, mover.sigma)
    assert not np.allclose(coords, mover.mesh.coordinates.dat.data, atol=tol)

    # Adapt to a constant monitor
    mover.monitor_function = const_monitor
    mover.move()
    if exports:
        File("outputs/const.pvd").write(mover.phi, mover.sigma)
    assert np.allclose(coords, mover.mesh.coordinates.dat.data, atol=tol)


@pytest.mark.slow
def test_bcs(mesh_dim, method, fix_boundary):
    """
    Test that domain boundaries are fixed by
    the Monge-Ampere movers.
    """
    n = 20
    mesh = mesh_dim(n)
    one = Constant(1.0)
    bnd = assemble(one * ds(domain=mesh))
    bnodes = DirichletBC(mesh.coordinates.function_space(), 0, "on_boundary").nodes
    bnd_coords = mesh.coordinates.dat.data.copy()[bnodes]

    # Adapt to a ring monitor
    mover = MongeAmpereMover(
        mesh, ring_monitor, method=method, fix_boundary_nodes=fix_boundary
    )
    mover.move()

    # Check boundary lengths are preserved
    bnd_new = assemble(one * ds(domain=mover.mesh))
    assert np.isclose(bnd, bnd_new)

    # Check boundaries are indeed fixed
    if fix_boundary:
        bnd_coords_new = mover.mesh.coordinates.dat.data[bnodes]
        assert np.allclose(bnd_coords, bnd_coords_new)
