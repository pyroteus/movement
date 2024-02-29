# Mesh movement using Laplacian smoothing
# =======================================
#
# In this demo, we demonstrate the *Laplacian smoothing* approach. This method relies on
# there being a user-specified forcing function, :math:`f`. With this, we can define a
# Poisson equation of the form
#
# .. math::
#    -\Delta\mathbf{v} = f
#
# where :math:`\mathbf{v}` is the so-called *mesh velocity* that we solve for. That is,
# with the mesh velocity, we update the mesh coordinates according to
#
# .. math::
#    \mathbf{u} := \mathbf{u} + \mathbf{v} * \Delta t,
#
# where :math:`\mathbf{u}` is the mesh coordinate field and :math:`\Delta t` is the
# timestep.
#
# We begin by importing from the namespaces of Firedrake and Movement. ::

from firedrake import *
from movement import *

# Let's start with a uniform mesh of the unit square. It has four boundary segments,
# which are tagged with the integers 1, 2, 3, and 4. Note that segment 4 corresponds to
# the top boundary. ::

import matplotlib.pyplot as plt
from firedrake.pyplot import triplot

n = 10
mesh = UnitSquareMesh(n, n)
coord_data_init = mesh.coordinates.dat.data.copy()
fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.set_aspect(1)
axes.legend()
plt.savefig("laplacian_smoothing-initial_mesh.jpg")

# .. figure:: laplacian_smoothing-initial_mesh.jpg
#    :figwidth: 50%
#    :align: center
#
# Suppose we wish to apply a time-dependent forcing to the top boundary and see how the
# mesh responds. Consider the forcing
#
# .. math::
#     \mathbf{f}(x,y,t)=\left[0, A\:\sin\left(\frac{2\pi t}T\right)\:\sin(\pi x)\right]
#
# acting only in the vertical direction, where :math:`A` is the amplitude and :math:`T`
# is the time period. The displacement following a sinusoidal pattern along that
# boundary. The forcing is also sinusoidal in time, such that it ramps up and then
# reverses, with the analytical solution being that the final mesh coincides with the
# initial mesh.

import numpy as np

forcing_period = 1.0
num_timesteps = 10
timestep = forcing_period / num_timesteps
forcing_amplitude = 50


def forcing(x, t):
    return forcing_amplitude * np.sin(2 * pi * t / forcing_period) * np.sin(pi * x)


X = np.linspace(0, 1, n + 1)
times = np.arange(0, forcing_period + 0.5 * timestep, timestep)

fig, axes = plt.subplots()
for t in times:
    axes.plot(X, forcing(X, t), label=f"t={t:.1f}")
axes.set_xlim([0, 1])
axes.legend()
box = axes.get_position()
axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axes.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("laplacian_smoothing-forcings.jpg")

# .. figure:: laplacian_smoothing-forcings.jpg
#    :figwidth: 60%
#    :align: center
#
# To apply this forcing, we need to create a :class:`~.LaplacianSmoother` instance and
# define a function for updating the forcing applied to the boundary nodes. Since we are
# going to apply a forcing to the top boundary, we need to extract the indices for the
# associated boundary nodes. This is done using a :class:`~.DirichletBC` object. ::

mover = LaplacianSmoother(mesh, timestep)
boundary_nodes = DirichletBC(mesh.coordinates.function_space(), 0, 4).nodes


def update_forcings(t):
    coord_data = mesh.coordinates.dat.data
    forcing_data = mover.f.dat.data
    for i in boundary_nodes:
        x, y = coord_data[i]
        forcing_data[i][1] = forcing(x, t)


# We are now able to apply the mesh movement method. The forcings effectively enforce a
# Dirichlet condition on the top boundary. On other boundaries, we enforce that there is
# no movement using the `fixed_boundaries` keyword argument. ::

import matplotlib.patches as mpatches

time = 0.0
fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(12, 10))
for i, t in enumerate(times):
    idx = 0 if i == 0 else (i + 1)

    # Move the mesh and calculate the displacement
    mover.move(t, update_forcings=update_forcings, fixed_boundaries=[1, 2, 3])
    displacement = np.linalg.norm(mover.displacement)
    print(f"time = {t:.1f} s, displacement = {displacement:.2f} m")

    # Plot the current mesh, adding a time label
    ax = axes[idx // 4, idx % 4]
    triplot(mover.mesh, axes=ax)
    ax.legend(handles=[mpatches.Patch(label=f"t={t:.1f}")], handlelength=0)
    ax.set_ylim([-0.05, 1.45])
axes[0, 1].axis(False)
plt.savefig("laplacian_smoothing-adapted_meshes.jpg")

# .. figure:: laplacian_smoothing-adapted_meshes.jpg
#    :figwidth: 100%
#    :align: center
#
# The mesh is deformed according to the vertical forcing, with the left, right, and
# bottom boundaries remaining fixed, returning it to be very close to its original state
# after one period. Let's check this in the :math:`\ell_\infty` norm. ::

coord_data = mover.mesh.coordinates.dat.data
linf_error = np.max(np.abs(coord_data - coord_data_init))
print(f"l_infinity error: {linf_error:.3f} m")
assert np.isclose(linf_error, 0.0)

# This tutorial can be downloaded as a `Python script <laplacian_smoothing.py>`__.
