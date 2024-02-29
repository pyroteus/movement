from firedrake import *
from movement import *
import numpy as np
import os
import pytest


@pytest.fixture(params=["laplacian"])
def method(request):
    return request.param


@pytest.fixture(params=[100])
def num_timesteps(request):
    return request.param


def test_forced(method, num_timesteps, plot=False, test=True):
    """
    Test that a uniform mesh is moved as expected
    when only one of its boundaries is forced.
    """
    dt = 1.0 / num_timesteps

    # Set parameters
    n = 10  # Mesh resolution
    A = 0.5  # Forcing amplitude
    T = 1.0  # Forcing period

    # Construct mesh and mover
    mesh = SquareMesh(n, n, 2)
    V = mesh.coordinates.function_space()
    coords = mesh.coordinates.dat.data.copy()
    if method == "laplacian":
        mover = LaplacianSmoother(mesh, dt)

    def update_forcings(t):
        """
        Sinusoidal forcing on the top boundary.
        """
        for i in DirichletBC(V, 0, 4).nodes:
            mover.f.dat.data[i][1] += (
                A * np.sin(2 * np.pi * t / T) * np.sin(np.pi * coords[i][0])
            )

    # Move the mesh
    time = 0.0
    for j in range(num_timesteps):
        mover.move(time, update_forcings=update_forcings, fixed_boundaries=[1, 2, 3])
        time += dt
    new_coords = mover.mesh.coordinates.dat.data

    # Plotting
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots()
        triplot(mover.mesh, axes=axes)
        axes.axis(False)
        plt.tight_layout()
        plt.savefig(f"plots/mesh_{method}.png")

    # Check as expected
    if test:
        pwd = os.path.dirname(__file__)
        fname = os.path.join(pwd, "data", "forced_mesh_laplacian.npy")
        assert np.allclose(new_coords, np.load(fname))


if __name__ == "__main__":
    mesh = test_forced("laplacian", 100, plot=True, test=False).mesh
    np.save("data/forced_mesh_laplacian", mesh.coordinates.dat.data)
