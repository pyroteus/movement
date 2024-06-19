import unittest
from unittest.mock import MagicMock

import numpy as np
import ufl
from monitors import *
from parameterized import parameterized

from movement import *


class BaseClasses:
    class TestMongeAmpere(unittest.TestCase):
        """
        Base class for Monge-ampere unit tests.
        """

        def mesh(self, dim=1, n=10, periodic=False):
            self.assertTrue(dim in (1, 2, 3))
            return {
                1: (PeriodicUnitIntervalMesh if periodic else UnitIntervalMesh)(n),
                2: (PeriodicUnitSquareMesh if periodic else UnitSquareMesh)(n, n),
                3: (PeriodicUnitCubeMesh if periodic else UnitCubeMesh)(n, n, n),
            }[dim]

        @property
        def dummy_mesh(self):
            return MagicMock(UnitSquareMesh)

        @property
        def dummy_monitor(self):
            return lambda *args: MagicMock(Function)


class TestExceptions(BaseClasses.TestMongeAmpere):
    """
    Unit tests for exceptions raised by Monge-Ampere movers.
    """

    def test_method_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            MongeAmpereMover(self.dummy_mesh, self.dummy_monitor, method="method")
        self.assertEqual(str(cm.exception), "Method 'method' not recognised.")

    def test_no_monitor_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            MongeAmpereMover(self.dummy_mesh, None)
        self.assertEqual(str(cm.exception), "Please supply a monitor function.")

    @parameterized.expand([("relaxation"), ("quasi_newton")])
    def test_maxiter_convergenceerror(self, method):
        """
        Test that the mesh mover raises a :class:`~.ConvergenceError` if it reaches the
        maximum number of iterations.
        """
        mesh = self.mesh(dim=2, n=4)
        mover = MongeAmpereMover(mesh, ring_monitor, method=method, maxiter=1)
        with self.assertRaises(ConvergenceError) as cm:
            mover.move()
        self.assertEqual(str(cm.exception), "Solver failed to converge in 1 iteration.")

    @parameterized.expand([(True,), (False,)])
    def test_divergence_convergenceerror(self, raise_errors):
        """
        Test that divergence of the mesh mover raises a :class:`~.ConvergenceError` if
        `raise_errors=True` and a :class:`~.Warning` otherwise.
        """
        mesh = self.mesh(dim=2, n=4)
        mover = MongeAmpereMover_Relaxation(
            mesh, ring_monitor, dtol=1.0e-08, raise_convergence_errors=raise_errors
        )
        with self.assertRaises(ConvergenceError if raise_errors else Warning) as cm:
            mover.move()
        self.assertEqual(str(cm.exception), "Solver diverged after 1 iteration.")

    def test_initial_guess_valueerror(self):
        mesh = self.mesh(n=2)
        phi_init = Function(FunctionSpace(mesh, "CG", 1))
        with self.assertRaises(ValueError) as cm:
            MongeAmpereMover_Relaxation(mesh, ring_monitor, phi_init=phi_init)
        self.assertEqual(str(cm.exception), "Need to initialise both phi *and* sigma.")

    def test_non_straight_boundary_valueerror(self):
        mesh = self.mesh(dim=2, n=3)
        mesh.coordinates.dat.data_with_halos[1][0] -= 0.25
        mover = MongeAmpereMover_Relaxation(mesh, ring_monitor)
        with self.assertRaises(ValueError) as cm:
            mover.move()
        msg = "Boundary segment '1' is not linear."
        self.assertEqual(str(cm.exception), msg)

    def test_non_flat_plane_valueerror(self):
        mesh = self.mesh(dim=3, n=3)
        mesh.coordinates.dat.data_with_halos[1][0] -= 0.25
        mover = MongeAmpereMover_Relaxation(mesh, ring_monitor)
        with self.assertRaises(ValueError) as cm:
            mover.move()
        msg = "Boundary segment '1' is not planar."
        self.assertEqual(str(cm.exception), msg)

    def test_invalid_plane_valuerror(self):
        mesh = self.mesh(dim=3, n=1)
        mesh.coordinates.dat.data_with_halos[:][:] = 0.0
        mover = MongeAmpereMover_Relaxation(mesh, ring_monitor)
        with self.assertRaises(ValueError) as cm:
            mover.move()
        msg = "Could not determine a plane for boundary segment '1'."
        self.assertEqual(str(cm.exception), msg)

    def test_periodic_fix_boundary_valueerror(self):
        mesh = self.mesh(n=3, periodic=True)
        mover = MongeAmpereMover_Relaxation(mesh, ring_monitor, fix_boundary_nodes=True)
        with self.assertRaises(ValueError) as cm:
            mover.move()
        msg = "Cannot fix boundary nodes for periodic meshes."
        self.assertEqual(str(cm.exception), msg)


class TestMonitor(BaseClasses.TestMongeAmpere):
    """
    Unit tests for monitor functions used by Monge-Ampere movers.
    """

    @parameterized.expand(
        [
            (1, "relaxation"),
            (1, "quasi_newton"),
            (2, "relaxation"),
            (2, "quasi_newton"),
            (3, "relaxation"),
            (3, "quasi_newton"),
        ]
    )
    def test_uniform_monitor(self, dim, method):
        """
        Test that the mesh mover converges in one iteration for a constant monitor
        function.
        """
        mesh = self.mesh(dim=dim)
        coords = mesh.coordinates.dat.data.copy()

        mover = MongeAmpereMover(
            mesh,
            const_monitor,
            method=method,
            phi_init=Function(FunctionSpace(mesh, "CG", 1)),
            sigma_init=Function(TensorFunctionSpace(mesh, "CG", 1)),
            rtol=1e-3,
        )
        num_iterations = mover.move()

        self.assertTrue(np.allclose(coords, mover.mesh.coordinates.dat.data))
        self.assertEqual(num_iterations, 0)

    @parameterized.expand(
        [(2, "relaxation"), (2, "quasi_newton"), (3, "relaxation"), (3, "quasi_newton")]
    )
    def test_change_monitor(self, dim, method):
        """
        Test that the mover can handle changes to the monitor function, such as would
        happen during timestepping.
        """
        mesh = self.mesh(dim=dim)
        gdim = mesh.geometric_dimension()
        coords = mesh.coordinates.dat.data.copy()
        atol = 1.0e-03
        rtol = 1.0e02 * atol**gdim

        # Adapt to a ring monitor
        mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=rtol)
        mover.move()
        moved_coords = mover.mesh.coordinates.dat.data
        self.assertFalse(np.allclose(coords, moved_coords, atol=atol))

        # Adapt to a constant monitor
        mover.monitor_function = const_monitor
        mover.move()
        moved_coords = mover.mesh.coordinates.dat.data
        self.assertTrue(np.allclose(coords, moved_coords, atol=atol))


class TestBCs(BaseClasses.TestMongeAmpere):
    """
    Unit tests for boundary conditions of Monge-Ampere movers.
    """

    def _test_boundary_preservation(self, mesh, method, fix_boundary):
        bnd = assemble(Constant(1.0) * ds(domain=mesh))
        bnodes = DirichletBC(mesh.coordinates.function_space(), 0, "on_boundary").nodes
        bnd_coords = mesh.coordinates.dat.data.copy()[bnodes]

        # Adapt to a ring monitor
        mover = MongeAmpereMover(
            mesh,
            ring_monitor,
            method=method,
            fix_boundary_nodes=fix_boundary,
            rtol=1e-3,
        )
        mover.move()

        # Check boundary lengths are preserved
        bnd_new = assemble(Constant(1.0) * ds(domain=mover.mesh))
        self.assertAlmostEqual(bnd, bnd_new)

        # Check boundaries are indeed fixed
        if fix_boundary:
            bnd_coords_new = mover.mesh.coordinates.dat.data[bnodes]
            self.assertTrue(np.allclose(bnd_coords, bnd_coords_new))
        return mover

    @parameterized.expand(
        [
            (1, "relaxation"),
            (1, "quasi_newton"),
            (2, "relaxation"),
            (2, "quasi_newton"),
            (3, "relaxation"),
            (3, "quasi_newton"),
        ]
    )
    def test_periodic(self, dim, method):
        """
        Test that periodic unit domains are not given boundary conditions by the
        Monge-Ampere movers.
        """
        mesh = self.mesh(dim=dim, periodic=True)
        volume = assemble(Constant(1.0) * dx(domain=mesh))
        mover = self._test_boundary_preservation(mesh, method, False)

        # Check the volume of the domain is conserved
        self.assertAlmostEqual(assemble(Constant(1.0) * dx(domain=mover.mesh)), volume)

        # Check that the variational problem does not have boundary conditions
        self.assertTrue(len(mover._l2_projector._problem.bcs) == 0)

    @parameterized.expand(
        [
            (1, "relaxation", True),
            (1, "relaxation", False),
            (1, "quasi_newton", True),
            (1, "quasi_newton", False),
            (2, "relaxation", True),
            (2, "relaxation", False),
            (2, "quasi_newton", True),
            (2, "quasi_newton", False),
            (3, "relaxation", True),
            (3, "relaxation", False),
            (3, "quasi_newton", True),
            (3, "quasi_newton", False),
        ]
    )
    def test_boundary_preservation_axis_aligned(self, dim, method, fix_boundary):
        """
        Test that boundaries of unit domains are preserved by the Monge-Ampere movers.
        """
        mesh = self.mesh(dim=dim)
        volume = assemble(Constant(1.0) * dx(domain=mesh))
        mover = self._test_boundary_preservation(mesh, method, fix_boundary)

        # Check the volume of the domain is conserved
        self.assertAlmostEqual(assemble(Constant(1.0) * dx(domain=mover.mesh)), volume)

        # Check that the variational problem has one DirichletBC per boundary segment
        bcs = mover._l2_projector._problem.bcs
        self.assertTrue(len(bcs) == 2 * dim)
        self.assertTrue(all(isinstance(bc, DirichletBC) for bc in bcs))

    @parameterized.expand(
        [
            (2, "relaxation", True),
            (2, "relaxation", False),
            (2, "quasi_newton", True),
            (2, "quasi_newton", False),
            (3, "relaxation", True),
            (3, "relaxation", False),
            (3, "quasi_newton", True),
            (3, "quasi_newton", False),
        ]
    )
    def test_boundary_preservation_non_axis_aligned(self, dim, method, fix_boundary):
        """
        Test that boundaries of rotated unit domains are preserved by the
        Monge-Ampere movers.
        """
        mesh = self.mesh(dim=dim)
        volume = assemble(Constant(1.0) * dx(domain=mesh))

        # Construct a new mesh by rotating by 45 degrees in the xy-plane
        cs = ufl.cos(ufl.pi / 4)
        sn = ufl.sin(ufl.pi / 4)
        if dim == 2:
            rotation_matrix = ufl.as_matrix([[cs, sn], [-sn, cs]])
        elif dim == 3:
            rotation_matrix = ufl.as_matrix([[cs, sn, 0], [-sn, cs, 0], [0, 0, 1]])
        else:
            raise ValueError(f"Dimension {dim} not supported.")
        coords = Function(mesh.coordinates.function_space())
        coords.interpolate(ufl.dot(rotation_matrix, mesh.coordinates))
        mover = self._test_boundary_preservation(Mesh(coords), method, fix_boundary)

        # Check the volume of the domain is conserved
        self.assertAlmostEqual(assemble(Constant(1.0) * dx(domain=mover.mesh)), volume)

        # If boundaries are not fixed then EquationBCs should be used for boundaries of
        # the xy-plane
        bcs = mover._l2_projector._problem.bcs
        if fix_boundary:
            self.assertTrue(len(bcs) == 2 * dim)
            self.assertTrue(all(isinstance(bc, DirichletBC) for bc in bcs))
        elif dim == 2:
            self.assertTrue(len(bcs) == 8)
            self.assertTrue(all(isinstance(bc, EquationBC) for bc in bcs))
        else:
            self.assertTrue(len(bcs) == 10)
            self.assertEqual(sum(isinstance(bc, EquationBC) for bc in bcs), 8)
            self.assertEqual(sum(isinstance(bc, DirichletBC) for bc in bcs), 2)


class TestMisc(BaseClasses.TestMongeAmpere):
    """
    Unit tests for other misc. functionality of Monge-Ampere movers.
    """

    @parameterized.expand(
        [
            (1, "relaxation"),
            (1, "quasi_newton"),
            (2, "relaxation"),
            (2, "quasi_newton"),
            (3, "relaxation"),
            (3, "quasi_newton"),
        ]
    )
    def test_continue(self, dim, method):
        """
        Test that providing a good initial guess benefits the solver.
        """
        mesh = self.mesh(dim=dim)
        rtol = 1.0e-03

        # Solve the problem to a weak tolerance
        mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=0.1)
        num_it_init = mover.move()

        # Continue with a tighter tolerance
        mover.rtol = rtol
        num_it_continue = mover.move()

        # Solve the problem again to a tight tolerance
        mesh = self.mesh(dim)
        mover = MongeAmpereMover(mesh, ring_monitor, method=method, rtol=rtol)
        num_it_naive = mover.move()

        self.assertLessEqual(num_it_continue, num_it_naive)
        self.assertLessEqual(num_it_init + num_it_continue, num_it_naive)
        # FIXME: Looks like the mesh is tangled or close to tangling
        #        for the relaxation method, which is concerning. (#4)

    @parameterized.expand(
        [
            (1, "relaxation"),
            (1, "quasi_newton"),
            (2, "relaxation"),
            (2, "quasi_newton"),
            (3, "relaxation"),
            (3, "quasi_newton"),
        ]
    )
    def test_coordinate_update(self, dim, method):
        mesh = self.mesh(dim=dim, n=2)
        dg_coords = Function(VectorFunctionSpace(mesh, "DG", 1))
        dg_coords.project(mesh.coordinates)
        mover = MongeAmpereMover(Mesh(dg_coords), const_monitor, method=method)
        mover._grad_phi.assign(2.0)
        mover._update_coordinates()
        self.assertNotEqual(mover.grad_phi.dof_dset.size, mover._grad_phi.dof_dset.size)
        self.assertAlmostEqual(errornorm(mover.grad_phi, mover._grad_phi), 0)
        mover.xi += 2
        self.assertAlmostEqual(errornorm(mover.x, mover.xi), 0)
