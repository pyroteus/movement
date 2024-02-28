import firedrake
from firedrake.petsc import PETSc
from pyadjoint import no_annotations
import ufl
import numpy as np
import movement.solver_parameters as solver_parameters
from movement.mover import PrimeMover
from movement.monitor_smoother import laplacian_smoothing


__all__ = [
    "MongeAmpereMover_Relaxation",
    "MongeAmpereMover_QuasiNewton",
    "MongeAmpereMover",
    "monge_ampere",
]


def MongeAmpereMover(mesh, monitor_function, method="relaxation", **kwargs):
    r"""
        Movement of a `mesh` is determined by a `monitor_function`
        :math:`m` and the Monge-Ampère type equation

    ..  math::
            m(x)\det(I + H(\phi)) = \theta,

        for a scalar potential :math:`\phi`, where :math:`I` is the
        identity matrix, :math:`\theta` is a normalisation coefficient
        and :math:`H(\phi)` denotes the Hessian of :math:`\phi` with
        respect to the coordinates :math:`\xi` of the computational mesh.

        The physical mesh coordinates :math:`x` are updated according to

    ..  math::
            x = \xi + \nabla\phi.

        :arg mesh: the physical mesh
        :arg monitor_function: a Python function which takes a mesh as input
        :kwarg method: choose from 'relaxation' and 'quasi_newton'
        :kwarg phi_init: initial guess for the scalar potential
        :kwarg sigma_init: initial guess for the Hessian
        :return: converged scalar potential and Hessian
    """
    if method == "relaxation":
        return MongeAmpereMover_Relaxation(mesh, monitor_function, **kwargs)
    elif method == "quasi_newton":
        return MongeAmpereMover_QuasiNewton(mesh, monitor_function, **kwargs)
    else:
        raise ValueError(f"Method {method} not recognised.")


class MongeAmpereMover_Base(PrimeMover):
    """
    Base class for mesh movers based on the solution
    of Monge-Ampere type equations.
    """

    def __init__(self, mesh, monitor_function, **kwargs):
        """
        :arg mesh: the physical mesh
        :arg monitor_function: a Python function which takes a mesh as input
        :kwarg phi_init: initial guess for the scalar potential
        :kwarg sigma_init: initial guess for the Hessian
        :kwarg maxiter: maximum number of iterations for the relaxation
        :kwarg rtol: relative tolerance for the residual
        :kwarg dtol: divergence tolerance for the residual
        :kwarg fix_boundary_nodes: should all boundary nodes remain fixed?
        """
        if monitor_function is None:
            raise ValueError("Please supply a monitor function")

        # Collect parameters before calling super
        self.pseudo_dt = firedrake.Constant(kwargs.pop("pseudo_timestep", 0.1))
        self.maxiter = kwargs.pop("maxiter", 1000)
        self.rtol = kwargs.pop("rtol", 1.0e-08)
        self.dtol = kwargs.pop("dtol", 2.0)
        self.fix_boundary_nodes = kwargs.pop("fix_boundary_nodes", False)
        self.smooth_monitor_values = kwargs.pop("smooth_monitor_values", False)
        self.monitor_smoother = kwargs.pop("monitor_smoother", "laplacian")
        self.laplacian_smoother_num = kwargs.pop("laplacian_smoother_num", 40)
        super().__init__(mesh, monitor_function=monitor_function)

        # Create function spaces
        self.P0 = firedrake.FunctionSpace(self.mesh, "DG", 0)
        self.P1 = firedrake.FunctionSpace(self.mesh, "CG", 1)
        self.P1_vec = firedrake.VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = firedrake.TensorFunctionSpace(self.mesh, "CG", 1)

        # Create objects used during the mesh movement
        self.theta = firedrake.Constant(0.0)
        self.monitor = firedrake.Function(self.P1, name="Monitor function")
        self.monitor.interpolate(self.monitor_function(self.mesh))
        self.volume = firedrake.Function(self.P0, name="Mesh volume")
        self.volume.interpolate(ufl.CellVolume(self.mesh))
        self.original_volume = firedrake.Function(self.volume)
        self.total_volume = firedrake.assemble(firedrake.Constant(1.0) * self.dx)
        self.L_P0 = firedrake.TestFunction(self.P0) * self.monitor * self.dx
        self._grad_phi = firedrake.Function(self.P1_vec)
        self.grad_phi = firedrake.Function(self.mesh.coordinates)

    @PETSc.Log.EventDecorator("MongeAmpereBase.apply_initial_guess")
    def apply_initial_guess(self, phi_init=None, sigma_init=None, **kwargs):
        """
        Initialise the approximations to the scalar potential
        and its hessian with an initial guess.

        By default, both are initialised to zero, which corresponds
        to the case where the computational and physical meshes
        coincide.

        :kwarg phi_init: initial guess for the scalar potential
        :kwarg sigma_init: initial guess for the Hessian
        """
        if phi_init is not None and sigma_init is not None:
            assert hasattr(self, "phi")
            assert hasattr(self, "sigma")
            assert hasattr(self, "phi_old")
            assert hasattr(self, "sigma_old")
            self.phi.assign(phi_init)
            self.sigma.assign(sigma_init)
            self.phi_old.assign(phi_init)
            self.sigma_old.assign(sigma_init)
        elif phi_init is not None or sigma_init is not None:
            raise ValueError("Need to initialise both phi *and* sigma")

    @property
    @PETSc.Log.EventDecorator()
    def _diagnostics(self):
        """
        Compute the following diagnostics:
          1) the ratio of the smallest and largest element volumes;
          2) coefficient of variation (σ/μ) of element volumes;
          3) relative L2 norm residual.
        """
        v = self.volume.vector().gather()
        minmax = v.min() / v.max()
        mean = v.sum() / v.max()
        w = v.copy() - mean
        w *= w
        std = np.sqrt(w.sum() / w.size)
        cv = std / mean
        assert hasattr(self, "_residual_l2_form")
        assert hasattr(self, "_norm_l2_form")
        residual_l2 = firedrake.assemble(self._residual_l2_form).dat.norm
        norm_l2 = firedrake.assemble(self._norm_l2_form).dat.norm
        residual_l2_rel = residual_l2 / norm_l2
        return minmax, residual_l2_rel, cv

    @property
    @PETSc.Log.EventDecorator("MongeAmpereBase.update_coordinates")
    def x(self):
        """
        Update the coordinate :class:`Function` using
        the recovered gradient.
        """
        try:
            self.grad_phi.assign(self._grad_phi)
        except Exception:
            self.grad_phi.interpolate(self._grad_phi)
        self._x.assign(self.xi + self.grad_phi)  # x = ξ + grad(φ)
        return self._x

    @property
    @PETSc.Log.EventDecorator("MongeAmpereBase.create_l2_projector")
    def l2_projector(self):
        """
        Create a linear solver for obtaining the gradient
        of the potential using an L2 projection.

        Boundary conditions are imposed as a post-processing step.
        """
        if hasattr(self, "_l2_projector"):
            return self._l2_projector
        assert hasattr(self, "phi_old")
        u_cts = firedrake.TrialFunction(self.P1_vec)
        v_cts = firedrake.TestFunction(self.P1_vec)

        # Domain interior
        a = ufl.inner(v_cts, u_cts) * self.dx
        L = ufl.inner(v_cts, ufl.grad(self.phi_old)) * self.dx

        # Enforce no movement normal to boundary
        n = ufl.FacetNormal(self.mesh)
        bcs = []
        for i in self.mesh.exterior_facets.unique_markers:
            if self.fix_boundary_nodes:
                bcs.append(firedrake.DirichletBC(self.P1_vec, 0, i))
                continue

            # Check for axis-aligned boundaries
            _n = [firedrake.assemble(abs(n[j]) * self.ds(i)) for j in range(self.dim)]
            if np.allclose(_n, 0.0):
                raise ValueError(f"Invalid normal vector {_n}")
            else:
                if self.dim != 2:
                    raise NotImplementedError  # TODO
                if np.isclose(_n[0], 0.0):
                    bcs.append(firedrake.DirichletBC(self.P1_vec.sub(1), 0, i))
                    continue
                elif np.isclose(_n[1], 0.0):
                    bcs.append(firedrake.DirichletBC(self.P1_vec.sub(0), 0, i))
                    continue

            # Enforce no mesh movement normal to boundaries
            a_bc = ufl.dot(v_cts, n) * ufl.dot(u_cts, n) * self.ds
            L_bc = ufl.dot(v_cts, n) * firedrake.Constant(0.0) * self.ds
            bcs.append(firedrake.EquationBC(a_bc == L_bc, self._grad_phi, i))

            # Allow tangential movement, but only up until the end of boundary segments
            s = ufl.perp(n)
            a_bc = ufl.dot(v_cts, s) * ufl.dot(u_cts, s) * self.ds
            L_bc = ufl.dot(v_cts, s) * ufl.dot(ufl.grad(self.phi_old), s) * self.ds
            edges = set(self.mesh.exterior_facets.unique_markers)
            if len(edges) == 0:
                bbc = None  # Periodic case
            else:
                from warnings import warn

                warn(
                    "Have you checked that all straight line segments are uniquely tagged?"
                )
                corners = [(i, j) for i in edges for j in edges.difference([i])]
                bbc = firedrake.DirichletBC(self.P1_vec, 0, corners)
            bcs.append(firedrake.EquationBC(a_bc == L_bc, self._grad_phi, i, bcs=bbc))

        # Create solver
        problem = firedrake.LinearVariationalProblem(a, L, self._grad_phi, bcs=bcs)
        sp = {
            "ksp_type": "cg",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }
        self._l2_projector = firedrake.LinearVariationalSolver(
            problem, solver_parameters=sp
        )
        return self._l2_projector


class MongeAmpereMover_Relaxation(MongeAmpereMover_Base):
    r"""
        The elliptic Monge-Ampere equation is solved in a parabolised
        form using a pseudo-time relaxation,

    ..  math::
            -\frac\partial{\partial\tau}\Delta\phi = m(x)\det(I + H(\phi)) - \theta,

        where :math:`\tau` is the pseudo-time variable. Forward Euler is
        used for the pseudo-time integration (see McRae et al. 2018 for
        details).

        References
        ==========
        A. T. T. McRae, C. J. Cotter, C. J. Budd, Optimal-transport-based
        mesh adaptivity on the plane and sphere using finite elements, SIAM
        Journal on Scientific Computing 40 (2) (2018) 1121–1148.
        doi:10.1137/16M1109515.
    """

    @PETSc.Log.EventDecorator("MongeAmpereMover.__init__")
    def __init__(self, mesh, monitor_function, **kwargs):
        """
        :arg mesh: the physical mesh
        :arg monitor_function: a Python function which takes a mesh as input
        :kwarg phi_init: initial guess for the scalar potential
        :kwarg sigma_init: initial guess for the Hessian
        :kwarg pseudo_timestep: pseudo-timestep to use for the relaxation
        :kwarg maxiter: maximum number of iterations for the relaxation
        :kwarg rtol: relative tolerance for the residual
        :kwarg dtol: divergence tolerance for the residual
        """
        self.pseudo_dt = firedrake.Constant(kwargs.pop("pseudo_timestep", 0.1))
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)

        # Create functions to hold solution data
        self.phi = firedrake.Function(self.P1)
        self.sigma = firedrake.Function(self.P1_ten)
        self.phi_old = firedrake.Function(self.P1)
        self.sigma_old = firedrake.Function(self.P1_ten)

        # Initialise phi and sigma
        self.apply_initial_guess(**kwargs)

        # Setup residuals
        I = ufl.Identity(self.dim)
        self.theta_form = self.monitor * ufl.det(I + self.sigma_old) * self.dx
        self.residual = self.monitor * ufl.det(I + self.sigma_old) - self.theta
        psi = firedrake.TestFunction(self.P1)
        self._residual_l2_form = psi * self.residual * self.dx
        self._norm_l2_form = psi * self.theta * self.dx

    @property
    @PETSc.Log.EventDecorator("MongeAmpereMover.create_pseudotimestepper")
    def pseudotimestepper(self):
        """
        Setup the pseudo-timestepper for the relaxation method.
        """
        if hasattr(self, "_pseudotimestepper"):
            return self._pseudotimestepper
        assert hasattr(self, "phi")
        assert hasattr(self, "phi_old")
        phi = firedrake.TrialFunction(self.P1)
        psi = firedrake.TestFunction(self.P1)
        a = ufl.inner(ufl.grad(psi), ufl.grad(phi)) * self.dx
        L = (
            ufl.inner(ufl.grad(psi), ufl.grad(self.phi_old)) * self.dx
            + self.pseudo_dt * psi * self.residual * self.dx
        )
        problem = firedrake.LinearVariationalProblem(a, L, self.phi)
        sp = {
            "ksp_type": "cg",
            "pc_type": "gamg",
        }
        nullspace = firedrake.VectorSpaceBasis(constant=True)
        self._pseudotimestepper = firedrake.LinearVariationalSolver(
            problem,
            solver_parameters=sp,
            nullspace=nullspace,
            transpose_nullspace=nullspace,
        )
        return self._pseudotimestepper

    @property
    @PETSc.Log.EventDecorator("MongeAmpereMover.create_equidistributor")
    def equidistributor(self):
        """
        Setup the equidistributor for the relaxation method.
        """
        if hasattr(self, "_equidistributor"):
            return self._equidistributor
        assert hasattr(self, "phi")
        assert hasattr(self, "sigma")
        if self.dim != 2:
            raise NotImplementedError  # TODO
        n = ufl.FacetNormal(self.mesh)
        sigma = firedrake.TrialFunction(self.P1_ten)
        tau = firedrake.TestFunction(self.P1_ten)
        a = ufl.inner(tau, sigma) * self.dx
        L = (
            -ufl.dot(ufl.div(tau), ufl.grad(self.phi)) * self.dx
            + (tau[0, 1] * n[1] * self.phi.dx(0) + tau[1, 0] * n[0] * self.phi.dx(1))
            * self.ds
        )
        problem = firedrake.LinearVariationalProblem(a, L, self.sigma)
        sp = {
            "ksp_type": "cg",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }
        self._equidistributor = firedrake.LinearVariationalSolver(
            problem, solver_parameters=sp
        )
        return self._equidistributor

    @PETSc.Log.EventDecorator("MongeAmpereMover.move")
    def move(self):
        """
        Run the relaxation method to convergence and update the mesh.
        """
        assert hasattr(self, "phi")
        assert hasattr(self, "sigma")
        assert hasattr(self, "phi_old")
        assert hasattr(self, "sigma_old")
        for i in range(self.maxiter):
            # L2 project
            self.l2_projector.solve()

            # Update mesh coordinates
            self.mesh.coordinates.assign(self.x)

            # Update monitor function
            self.monitor.interpolate(self.monitor_function(self.mesh))

            # Perform monitor function values smooth
            if self.smooth_monitor_values:
                if self.monitor_smoother == "laplacian":
                    self.monitor = laplacian_smoothing(
                        mesh=self.mesh,
                        monitor_values=self.monitor,
                        smooth_num=self.laplacian_smoother_num,
                    )
                else:
                    raise NotImplementedError(
                        f"Monitor smoother {self.monitor_smoother} not implemented."
                    )
            firedrake.assemble(self.L_P0, tensor=self.volume)
            self.volume.interpolate(self.volume / self.original_volume)
            self.mesh.coordinates.assign(self.xi)

            # Evaluate normalisation coefficient
            self.theta.assign(firedrake.assemble(self.theta_form) / self.total_volume)

            # Check convergence criteria
            minmax, residual, cv = self._diagnostics
            if i == 0:
                initial_norm = residual
            PETSc.Sys.Print(
                f"{i:4d}"
                f"   Min/Max {minmax:10.4e}"
                f"   Residual {residual:10.4e}"
                f"   Variation (σ/μ) {cv:10.4e}"
            )
            if residual < self.rtol:
                PETSc.Sys.Print(f"Converged in {i+1} iterations.")
                break
            if residual > self.dtol * initial_norm:
                raise firedrake.ConvergenceError(f"Diverged after {i+1} iterations.")
            if i == self.maxiter - 1:
                raise firedrake.ConvergenceError(
                    f"Failed to converge in {i+1} iterations."
                )

            # Apply pseudotimestepper and equidistributor
            self.pseudotimestepper.solve()
            self.equidistributor.solve()
            self.phi_old.assign(self.phi)
            self.sigma_old.assign(self.sigma)
        self.mesh.coordinates.assign(self.x)
        return i


class MongeAmpereMover_QuasiNewton(MongeAmpereMover_Base):
    r"""
    The elliptic Monge-Ampere equation is solved using a quasi-Newton
    method (see McRae et al. 2018 for details).

    References
    ==========
    A. T. T. McRae, C. J. Cotter, C. J. Budd, Optimal-transport-based
    mesh adaptivity on the plane and sphere using finite elements, SIAM
    Journal on Scientific Computing 40 (2) (2018) 1121–1148.
    doi:10.1137/16M1109515.
    """

    @PETSc.Log.EventDecorator("MongeAmpereMover.__init__")
    def __init__(self, mesh, monitor_function, **kwargs):
        """
        :arg mesh: the physical mesh
        :arg monitor_function: a Python function which takes a mesh as input
        :kwarg maxiter: maximum number of iterations for the relaxation
        :kwarg rtol: relative tolerance for the residual
        :kwarg dtol: divergence tolerance for the residual
        """
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)

        # Create functions to hold solution data
        self.V = self.P1 * self.P1_ten
        self.phisigma = firedrake.Function(self.V)
        self.phi, self.sigma = self.phisigma.subfunctions
        self.phisigma_old = firedrake.Function(self.V)
        self.phi_old, self.sigma_old = self.phisigma_old.subfunctions

        # Initialise phi and sigma
        self.apply_initial_guess(**kwargs)

        # Setup residuals
        I = ufl.Identity(self.dim)
        self.theta_form = self.monitor * ufl.det(I + self.sigma_old) * self.dx
        self.residual = self.monitor * ufl.det(I + self.sigma_old) - self.theta
        psi = firedrake.TestFunction(self.P1)
        self._residual_l2_form = psi * self.residual * self.dx
        self._norm_l2_form = psi * self.theta * self.dx

    @property
    @PETSc.Log.EventDecorator("MongeAmpereMover.create_equidistributor")
    def equidistributor(self):
        """
        Setup the equidistributor for the quasi-newton method.
        """
        if hasattr(self, "_equidistributor"):
            return self._equidistributor
        assert hasattr(self, "phisigma")
        if self.dim != 2:
            raise NotImplementedError  # TODO
        n = ufl.FacetNormal(self.mesh)
        I = ufl.Identity(self.dim)
        phi, sigma = firedrake.split(self.phisigma)
        psi, tau = firedrake.TestFunctions(self.V)
        F = (
            ufl.inner(tau, sigma) * self.dx
            + ufl.dot(ufl.div(tau), ufl.grad(phi)) * self.dx
            - (tau[0, 1] * n[1] * phi.dx(0) + tau[1, 0] * n[0] * phi.dx(1)) * self.ds
            - psi * (self.monitor * ufl.det(I + sigma) - self.theta) * self.dx
        )
        phi, sigma = firedrake.TrialFunctions(self.V)

        @PETSc.Log.EventDecorator("MongeAmpereMover.update_monitor")
        def update_monitor(cursol):
            """
            Callback for updating the monitor function.
            """
            assert hasattr(self, "phisigma_old")
            with self.phisigma_old.dat.vec as v:
                cursol.copy(v)
            self.l2_projector.solve()
            self.mesh.coordinates.assign(self.x)
            self.monitor.interpolate(self.monitor_function(self.mesh))
            self.mesh.coordinates.assign(self.xi)
            self.theta.assign(
                firedrake.assemble(self.theta_form) * self.total_volume ** (-1)
            )

        # Custom preconditioner
        Jp = (
            ufl.inner(tau, sigma) * self.dx
            + phi * psi * self.dx
            + ufl.inner(ufl.grad(phi), ufl.grad(psi)) * self.dx
        )

        # Setup the variational problem
        problem = firedrake.NonlinearVariationalProblem(F, self.phisigma, Jp=Jp)
        nullspace = firedrake.MixedVectorSpaceBasis(
            self.V, [firedrake.VectorSpaceBasis(constant=True), self.V.sub(1)]
        )
        sp = (
            solver_parameters.serial_qn
            if firedrake.COMM_WORLD.size == 1
            else solver_parameters.parallel_qn
        )
        sp["snes_atol"] = self.rtol
        sp["snes_max_it"] = self.maxiter
        self._equidistributor = firedrake.NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            transpose_nullspace=nullspace,
            pre_function_callback=update_monitor,
            pre_jacobian_callback=update_monitor,
            solver_parameters=sp,
        )

        @no_annotations
        @PETSc.Log.EventDecorator("MongeAmpereMover.monitor")
        def monitor(snes, i, rnorm):
            """
            Print progress of the optimisation to screen.

            Note that convergence is not actually checked.
            """
            cursol = snes.getSolution()
            update_monitor(cursol)
            self.mesh.coordinates.assign(self.x)
            firedrake.assemble(self.L_P0, tensor=self.volume)
            self.volume.interpolate(self.volume / self.original_volume)
            self.mesh.coordinates.assign(self.xi)
            minmax, residual, cv = self._diagnostics
            PETSc.Sys.Print(
                f"{i:4d}"
                f"   Min/Max {minmax:10.4e}"
                f"   Residual {residual:10.4e}"
                f"   Variation (σ/μ) {cv:10.4e}"
            )

        self.snes = self._equidistributor.snes
        self.snes.setMonitor(monitor)
        return self._equidistributor

    @PETSc.Log.EventDecorator("MongeAmpereMover.move")
    def move(self):
        """
        Run the quasi-Newton method to convergence and update the mesh.
        """
        try:
            self.equidistributor.solve()
            i = self.snes.getIterationNumber()
            PETSc.Sys.Print(f"Converged in {i} iterations.")
        except firedrake.ConvergenceError:
            i = self.snes.getIterationNumber()
            raise firedrake.ConvergenceError(f"Failed to converge in {i} iterations.")
        self.mesh.coordinates.assign(self.x)
        return i


def monge_ampere(mesh, monitor_function, method="relaxation", **kwargs):
    r"""
        Movement of a `mesh` is determined by a `monitor_function`
        :math:`m` and the Monge-Ampère type equation

    ..  math::
            m(x)\det(I + H(\phi)) = \theta,

        for a scalar potential :math:`\phi`, where :math:`I` is the
        identity matrix, :math:`\theta` is a normalisation coefficient
        and :math:`H(\phi)` denotes the Hessian of :math:`\phi` with
        respect to the coordinates :math:`\xi` of the computational mesh.

        The physical mesh coordinates :math:`x` are updated according to

    ..  math::
            x = \xi + \nabla\phi.

        :arg mesh: the physical mesh
        :arg monitor_function: a Python function which takes a mesh as input
        :kwarg method: choose from 'relaxation' and 'quasi_newton'
        :kwarg phi_init: initial guess for the scalar potential
        :kwarg sigma_init: initial guess for the Hessian
        :return: converged scalar potential and Hessian
    """
    if method == "relaxation":
        mover = MongeAmpereMover_Relaxation(mesh, monitor_function, **kwargs)
    elif method == "quasi_newton":
        mover = MongeAmpereMover_QuasiNewton(mesh, monitor_function, **kwargs)
    else:
        raise ValueError(f"Monge-Ampere solver {method} not recognised.")
    mover.adapt()
    return mover.phi, mover.sigma
