import abc
from warnings import warn

import firedrake
import firedrake.exceptions as fexc
import numpy as np
import ufl
from firedrake.petsc import PETSc
from pyadjoint import no_annotations

import movement.solver_parameters as solver_parameters
from movement.mover import PrimeMover

__all__ = [
    "MongeAmpereMover_Relaxation",
    "MongeAmpereMover_QuasiNewton",
    "MongeAmpereMover",
]


def tangential(v, n):
    """Return component of v perpendicular to n (assumed normalized)"""
    return v - ufl.dot(v, n) * n


def MongeAmpereMover(mesh, monitor_function, method="relaxation", **kwargs):
    r"""
    Movement of a `mesh` is determined by a `monitor_function`
    :math:`m` and the Monge-Ampère type equation

    .. math::
        m(x)\det(I + H(\phi)) = \theta,

    for a scalar potential :math:`\phi`, where :math:`I` is the
    identity matrix, :math:`\theta` is a normalisation coefficient
    and :math:`H(\phi)` denotes the Hessian of :math:`\phi` with
    respect to the coordinates :math:`\xi` of the computational mesh.

    The physical mesh coordinates :math:`x` are updated according to

    .. math::
        x = \xi + \nabla\phi.

    :arg mesh: the physical mesh
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :arg monitor_function: a Python function which takes a mesh as input
    :type monitor_function: :class:`~.Callable`
    :kwarg method: choose from 'relaxation' and 'quasi_newton'
    :type method: :class:`str`
    :kwarg phi_init: initial guess for the scalar potential
    :type phi_init: :class:`firedrake.function.Function`
    :kwarg sigma_init: initial guess for the Hessian
    :type sigma_init: :class:`firedrake.function.Function`
    :return: the Monge-Ampere Mover object
    :rtype: :class:`MongeAmpereMover_Relaxation` or
        :class:`MongeAmpereMover_QuasiNewton`
    """
    if method == "relaxation":
        return MongeAmpereMover_Relaxation(mesh, monitor_function, **kwargs)
    elif method == "quasi_newton":
        return MongeAmpereMover_QuasiNewton(mesh, monitor_function, **kwargs)
    else:
        raise ValueError(f"Method '{method}' not recognised.")


class MongeAmpereMover_Base(PrimeMover, metaclass=abc.ABCMeta):
    """
    Base class for mesh movers based on the solution
    of Monge-Ampere type equations.
    """

    def __init__(self, mesh, monitor_function, **kwargs):
        """
        :arg mesh: the physical mesh
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg monitor_function: a Python function which takes a mesh as input
        :type monitor_function: :class:`~.Callable`
        :kwarg phi_init: initial guess for the scalar potential
        :type phi_init: :class:`firedrake.function.Function`
        :kwarg sigma_init: initial guess for the Hessian
        :type sigma_init: :class:`firedrake.function.Function`
        :kwarg maxiter: maximum number of iterations for the relaxation
        :type maxiter: :class:`int`
        :kwarg rtol: relative tolerance for the residual
        :type rtol: :class:`float`
        :kwarg dtol: divergence tolerance for the residual
        :type dtol: :class:`float`
        :kwarg fix_boundary_nodes: should all boundary nodes remain fixed?
        :type fix_boundary_nodes: :class:`bool`
        """
        if monitor_function is None:
            raise ValueError("Please supply a monitor function.")

        # Collect parameters before calling super
        self.maxiter = kwargs.pop("maxiter", 1000)
        self.rtol = kwargs.pop("rtol", 1.0e-08)
        self.dtol = kwargs.pop("dtol", 2.0)
        self.fix_boundary_nodes = kwargs.pop("fix_boundary_nodes", False)
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)

        # Create function spaces
        self.P0 = firedrake.FunctionSpace(self.mesh, "DG", 0)
        self.P1 = firedrake.FunctionSpace(self.mesh, "CG", 1)
        self.P1_vec = firedrake.VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = firedrake.TensorFunctionSpace(self.mesh, "CG", 1)

        # Create objects used during the mesh movement
        self._create_functions()
        self.theta = firedrake.Constant(0.0)
        self.monitor.interpolate(self.monitor_function(self.mesh))
        self.volume.interpolate(ufl.CellVolume(self.mesh))
        self.original_volume = firedrake.Function(self.volume)
        self.total_volume = firedrake.assemble(firedrake.Constant(1.0) * self.dx)
        self.L_P0 = firedrake.TestFunction(self.P0) * self.monitor * self.dx

    @abc.abstractmethod
    def _create_functions(self):
        self.monitor = firedrake.Function(self.P1, name="Monitor function")
        self.volume = firedrake.Function(self.P0, name="Mesh volume")
        self._grad_phi = firedrake.Function(self.P1_vec)
        self.grad_phi = firedrake.Function(self.mesh.coordinates)

    @PETSc.Log.EventDecorator()
    def apply_initial_guess(self, phi_init, sigma_init):
        """
        Initialise the approximations to the scalar potential and its Hessian with an
        initial guess.

        By default, both are initialised to zero, which corresponds to the case where
        the computational and physical meshes coincide.

        :arg phi_init: initial guess for the scalar potential
        :type phi_init: :class:`firedrake.function.Function`
        :arg sigma_init: initial guess for the Hessian
        :type sigma_init: :class:`firedrake.function.Function`
        """
        if phi_init is not None and sigma_init is not None:
            self.phi.project(phi_init)
            self.sigma.project(sigma_init)
            self.phi_old.project(phi_init)
            self.sigma_old.project(sigma_init)
        elif phi_init is not None or sigma_init is not None:
            raise ValueError("Need to initialise both phi *and* sigma.")

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

    @PETSc.Log.EventDecorator()
    def _update_coordinates(self):
        r"""
        Update the physical coordinates :math:`\mathbf{x}` using the recovered gradient:
        .. math::
            \mathbf{x} = \boldsymbol{\xi} + \nabla_{\boldsymbol{\xi}}\phi.
        """
        try:
            self.grad_phi.assign(self._grad_phi)
        except Exception:
            self.grad_phi.interpolate(self._grad_phi)
        self.x.assign(self.xi + self.grad_phi)
        self.mesh.coordinates.assign(self.x)

        if hasattr(self, "tangling_checker"):
            self.tangling_checker.check()

    @property
    @PETSc.Log.EventDecorator()
    def l2_projector(self):
        """
        Create a linear solver for obtaining the gradient of the potential using an
        :math:`L^2` projection.

        Boundary conditions are imposed as a post-processing step.

        :return: the linear solver
        :rtype: :class:`~.LinearVariationalSolver`
        """
        if hasattr(self, "_l2_projector"):
            return self._l2_projector
        u_cts = firedrake.TrialFunction(self.P1_vec)
        v_cts = firedrake.TestFunction(self.P1_vec)

        # Domain interior
        a = ufl.inner(v_cts, u_cts) * self.dx
        L = ufl.inner(v_cts, ufl.grad(self.phi_old)) * self.dx

        # Enforce no movement normal to boundary
        n = ufl.FacetNormal(self.mesh)
        bcs = []
        for tag in self.mesh.exterior_facets.unique_markers:
            # TODO: Write tests for the boundary conditions block below (#79)
            if self.fix_boundary_nodes:
                bcs.append(firedrake.DirichletBC(self.P1_vec, 0, tag))
                continue

            # Check for axis-aligned boundaries
            _n = [firedrake.assemble(abs(n[j]) * self.ds(tag)) for j in range(self.dim)]
            iszero = [np.allclose(ni, 0.0) for ni in _n]
            nzero = sum(iszero)
            assert nzero < self.dim
            if nzero == self.dim - 1:
                idx = iszero.index(False)
                bcs.append(firedrake.DirichletBC(self.P1_vec.sub(idx), 0, tag))
                continue

            # Enforce no mesh movement normal to boundaries
            a_bc = ufl.dot(v_cts, n) * ufl.dot(u_cts, n) * self.ds
            L_bc = ufl.dot(v_cts, n) * firedrake.Constant(0.0) * self.ds
            bcs.append(firedrake.EquationBC(a_bc == L_bc, self._grad_phi, tag))

            # Allow tangential movement, but only up until the end of boundary segments
            a_bc = ufl.dot(tangential(v_cts, n), tangential(u_cts, n)) * self.ds
            L_bc = (
                ufl.dot(tangential(v_cts, n), tangential(ufl.grad(self.phi_old), n))
                * self.ds
            )
            edges = set(self.mesh.exterior_facets.unique_markers)
            if len(edges) == 0:
                bbc = None  # Periodic case
            else:
                warn(
                    "Have you checked that all straight line segments are uniquely"
                    " tagged?"
                )
                corners = [(i, j) for i in edges for j in edges.difference([i])]
                bbc = firedrake.DirichletBC(self.P1_vec, 0, corners)
            bcs.append(firedrake.EquationBC(a_bc == L_bc, self._grad_phi, tag, bcs=bbc))

        # Create solver
        problem = firedrake.LinearVariationalProblem(a, L, self._grad_phi, bcs=bcs)
        self._l2_projector = firedrake.LinearVariationalSolver(
            problem, solver_parameters=solver_parameters.cg_ilu
        )
        return self._l2_projector


class MongeAmpereMover_Relaxation(MongeAmpereMover_Base):
    r"""
    The elliptic Monge-Ampere equation is solved in a parabolised form using a
    pseudo-time relaxation,

    .. math::
        -\frac\partial{\partial\tau}\Delta\phi = m(x)\det(I + H(\phi)) - \theta,

    where :math:`\tau` is the pseudo-time variable. Forward Euler is used for the
    pseudo-time integration (see :cite:`MCB:18` for details).
    """

    @PETSc.Log.EventDecorator()
    def __init__(
        self, mesh, monitor_function, phi_init=None, sigma_init=None, **kwargs
    ):
        """
        :arg mesh: the physical mesh
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg monitor_function: a Python function which takes a mesh as input
        :type monitor_function: :class:`~.Callable`
        :kwarg phi_init: initial guess for the scalar potential
        :type phi_init: :class:`firedrake.function.Function`
        :kwarg sigma_init: initial guess for the Hessian
        :type sigma_init: :class:`firedrake.function.Function`
        :kwarg pseudo_timestep: pseudo-timestep to use for the relaxation
        :type pseudo_timestep: :class:`float`
        :kwarg maxiter: maximum number of iterations for the relaxation
        :type maxiter: :class:`int`
        :kwarg rtol: relative tolerance for the residual
        :type rtol: :class:`float`
        :kwarg dtol: divergence tolerance for the residual
        :type dtol: :class:`float`
        """
        self.pseudo_dt = firedrake.Constant(kwargs.pop("pseudo_timestep", 0.1))
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)

        # Initialise phi and sigma
        if phi_init or sigma_init:
            self.apply_initial_guess(phi_init, sigma_init)

        # Setup residuals
        I = ufl.Identity(self.dim)
        self.theta_form = self.monitor * ufl.det(I + self.sigma_old) * self.dx
        self.residual = self.monitor * ufl.det(I + self.sigma_old) - self.theta
        psi = firedrake.TestFunction(self.P1)
        self._residual_l2_form = psi * self.residual * self.dx
        self._norm_l2_form = psi * self.theta * self.dx

    def _create_functions(self):
        super()._create_functions()
        self.phi = firedrake.Function(self.P1)
        self.sigma = firedrake.Function(self.P1_ten)
        self.phi_old = firedrake.Function(self.P1)
        self.sigma_old = firedrake.Function(self.P1_ten)

    @property
    @PETSc.Log.EventDecorator()
    def pseudotimestepper(self):
        """
        Setup the pseudo-timestepper for the relaxation method.

        :return: the pseudo-timestepper
        :rtype: :class:`~.LinearVariationalSolver`
        """
        if hasattr(self, "_pseudotimestepper"):
            return self._pseudotimestepper
        phi = firedrake.TrialFunction(self.P1)
        psi = firedrake.TestFunction(self.P1)
        a = ufl.inner(ufl.grad(psi), ufl.grad(phi)) * self.dx
        L = (
            ufl.inner(ufl.grad(psi), ufl.grad(self.phi_old)) * self.dx
            + self.pseudo_dt * psi * self.residual * self.dx
        )
        problem = firedrake.LinearVariationalProblem(a, L, self.phi)
        nullspace = firedrake.VectorSpaceBasis(constant=True)
        self._pseudotimestepper = firedrake.LinearVariationalSolver(
            problem,
            solver_parameters=solver_parameters.cg_gamg,
            nullspace=nullspace,
            transpose_nullspace=nullspace,
        )
        return self._pseudotimestepper

    @property
    @PETSc.Log.EventDecorator()
    def equidistributor(self):
        """
        Setup the equidistributor for the relaxation method.

        :return: the equidistributor
        :rtype: :class:`~.LinearVariationalSolver`
        """
        if hasattr(self, "_equidistributor"):
            return self._equidistributor
        n = ufl.FacetNormal(self.mesh)
        sigma = firedrake.TrialFunction(self.P1_ten)
        tau = firedrake.TestFunction(self.P1_ten)
        a = ufl.inner(tau, sigma) * self.dx
        L = (
            -ufl.dot(ufl.div(tau), ufl.grad(self.phi)) * self.dx
            + ufl.dot(ufl.dot(tangential(ufl.grad(self.phi), n), tau), n) * self.ds
        )
        problem = firedrake.LinearVariationalProblem(a, L, self.sigma)
        self._equidistributor = firedrake.LinearVariationalSolver(
            problem, solver_parameters=solver_parameters.cg_ilu
        )
        return self._equidistributor

    @PETSc.Log.EventDecorator()
    def move(self):
        r"""
        Run the relaxation method to convergence and update the mesh.

        :return: the iteration count
        :rtype: :class:`int`
        """
        # Take iterations of the relaxed system until reaching convergence
        for i in range(self.maxiter):
            self.l2_projector.solve()
            self._update_coordinates()

            # Update monitor function
            self.monitor.interpolate(self.monitor_function(self.mesh))
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
                self._convergence_message(i + 1)
                break
            if residual > self.dtol * initial_norm:
                self._divergence_error(i + 1)
            if i == self.maxiter - 1:
                self._convergence_error(i + 1)

            # Apply pseudotimestepper and equidistributor
            self.pseudotimestepper.solve()
            self.equidistributor.solve()
            self.phi_old.assign(self.phi)
            self.sigma_old.assign(self.sigma)

        # Update mesh coordinates accordingly
        self._update_coordinates()
        return i


class MongeAmpereMover_QuasiNewton(MongeAmpereMover_Base):
    r"""
    The elliptic Monge-Ampere equation is solved using a quasi-Newton method (see
    :cite:`MCB:18` for details).
    """

    @PETSc.Log.EventDecorator()
    def __init__(
        self, mesh, monitor_function, phi_init=None, sigma_init=None, **kwargs
    ):
        """
        :arg mesh: the physical mesh
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        :arg monitor_function: a Python function which takes a mesh as input
        :type monitor_function: :class:`~.Callable`
        :kwarg phi_init: initial guess for the scalar potential
        :type phi_init: :class:`firedrake.function.Function`
        :kwarg sigma_init: initial guess for the Hessian
        :type sigma_init: :class:`firedrake.function.Function`
        :kwarg maxiter: maximum number of iterations for the Quasi-Newton solver
        :type maxiter: :class:`int`
        :kwarg rtol: relative tolerance for the residual
        :type rtol: :class:`float`
        :kwarg dtol: divergence tolerance for the residual
        :type dtol: :class:`float`
        """
        super().__init__(mesh, monitor_function=monitor_function, **kwargs)

        # Initialise phi and sigma
        if phi_init or sigma_init:
            self.apply_initial_guess(phi_init, sigma_init)

        # Setup residuals
        I = ufl.Identity(self.dim)
        self.theta_form = self.monitor * ufl.det(I + self.sigma_old) * self.dx
        self.residual = self.monitor * ufl.det(I + self.sigma_old) - self.theta
        psi = firedrake.TestFunction(self.P1)
        self._residual_l2_form = psi * self.residual * self.dx
        self._norm_l2_form = psi * self.theta * self.dx

    def _create_functions(self):
        super()._create_functions()
        self.V = self.P1 * self.P1_ten
        self.phisigma = firedrake.Function(self.V)
        self.phi, self.sigma = self.phisigma.subfunctions
        self.phisigma_old = firedrake.Function(self.V)
        self.phi_old, self.sigma_old = self.phisigma_old.subfunctions

    @property
    @PETSc.Log.EventDecorator()
    def equidistributor(self):
        """
        Setup the equidistributor for the quasi-newton method.

        :return: the equidistributor
        :rtype: :class:`~.NonlinearVariationalSolver`
        """
        if hasattr(self, "_equidistributor"):
            return self._equidistributor
        n = ufl.FacetNormal(self.mesh)
        I = ufl.Identity(self.dim)
        phi, sigma = firedrake.split(self.phisigma)
        psi, tau = firedrake.TestFunctions(self.V)
        F = (
            ufl.inner(tau, sigma) * self.dx
            + ufl.dot(ufl.div(tau), ufl.grad(phi)) * self.dx
            - ufl.dot(ufl.dot(tangential(ufl.grad(phi), n), tau), n) * self.ds
            - psi * (self.monitor * ufl.det(I + sigma) - self.theta) * self.dx
        )
        phi, sigma = firedrake.TrialFunctions(self.V)

        @PETSc.Log.EventDecorator("MongeAmpereMover.update_monitor")
        def update_monitor(cursol):
            """
            Callback for updating the monitor function.
            """
            with self.phisigma_old.dat.vec as v:
                cursol.copy(v)
            self.l2_projector.solve()
            self._update_coordinates()
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
            self._update_coordinates()
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

    @PETSc.Log.EventDecorator()
    def move(self):
        r"""
        Run the quasi-Newton method to convergence and update the mesh.

        :return: the iteration count
        :rtype: :class:`int`
        """
        # Solve equidistribution problem, handling convergence errors according to
        # desired behaviour
        try:
            self.equidistributor.solve()
            self._convergence_message(self.snes.getIterationNumber())
        except fexc.ConvergenceError as conv_err:
            self._convergence_error(self.snes.getIterationNumber(), exception=conv_err)

        # Update mesh coordinates accordingly
        self._update_coordinates()
        return self.snes.getIterationNumber()
