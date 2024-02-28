import firedrake as fd

__all__ = ["laplacian_smoothing"]


def laplacian_smoothing(mesh, monitor_values, smooth_num=40):
    """
    Laplacian smoothing for monitor function values

    Solve the equation below:
    m_{smooth} = dt * K * \nabla^2 m_{smooth} + m

    K = N * dx^2 / (4 * dt)
    N (smooth_num): the number of number of applications of a (1, −2, 1) filter

    Reference:
    Mariana C A Clare, Joseph Gregory Wallwork , Stephan C Kramer, Hilary Weller, Colin J Cotter, Matthew Piggott, On the use of mesh movement methods to help overcome the multi-scale
    challenges associated with hydro-morphodynamic modelling
    https://eartharxiv.org/repository/view/1751/


    """
    V = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    f = monitor_values
    dx = mesh.cell_sizes.dat.data[:].mean()
    K = smooth_num * dx**2 / 4
    RHS = f * v * fd.dx(domain=mesh)
    LHS = (K * fd.dot(fd.grad(v), fd.grad(u)) + v * u) * fd.dx(domain=mesh)
    bc = fd.DirichletBC(V, f, "on_boundary")

    monitor_smoothed = fd.Function(V)
    fd.solve(
        LHS == RHS,
        monitor_smoothed,
        solver_parameters={"ksp_type": "cg", "pc_type": "none"},
        bcs=bc,
    )
    return monitor_values.project(monitor_smoothed)
