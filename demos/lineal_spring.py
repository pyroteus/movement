# Mesh movement using the lineal spring approach
# ==============================================
#
# In this demo, we demonstrate a basic example using the *lineal spring* method, as
# described in :cite:`Farhat:1998`. For simplicity of presentation, we consider the same
# example considered in the `Laplacian smoothing <laplacian_smoothing.py.html>`__ demo,
# where mesh movement is driven by a forcing on the top boundary of a square mesh.
#
# The idea of the lineal spring method is to re-interpret the edges of a mesh as a
# structure of stiff beams. Each beam has a stiffness associated with it, which is
# related to its length and its orientation. We can assemble this information as a
# *stiffness matrix*,
#
# .. math::
#     \underline{\mathbf{K}} = \begin{bmatrix}
#     \underline{\mathbf{K}_{11}} && \dots && \underline{\mathbf{K}_{1N}}\\
#     \vdots && \ddots && \vdots\\
#     \underline{\mathbf{K}_{N1}} && \dots && \underline{\mathbf{K}_{NN}}\\
#     \end{bmatrix},
#
# where :math:`N` is the number of vertices in the mesh and each block
# :math:`\underline{\mathbf{K}_{ij}}` is a zero matrix if and only if vertex :math:`i`
# is *not* connected to vertex :math:`j`. For a 2D problem, each
# :math:`\underline{\mathbf{K}_{ij}}\in\mathbb{R}^{2\times2}` and
# :math:`\underline{\mathbf{K}}\in\mathbb{R}^{2N\times2N}`.
#
# As with the Laplacian smoothing method, the lineal spring approach relies on there
# being a user-specified forcing function. This acts on the vertices according to a
# forcing matrix,
#
# .. math::
#     \underline{\mathbf{f}} = \begin{bmatrix}
#         \mathbf{f}_1\\
#         \vdots\\
#         \mathbf{f}_N\\
#     \end{bmatrix}
#     \in\mathbb{R}^{N\times2}.
#
# That is, vertex :math:`i` is forced according to a vector :math:`\mathbf{f}_i`. Then
# we are able to compute the displacement of the vertices by solving the linear system
#
# .. math::
#     \underline{\mathbf{K}} \mathbf{u} = \mathbf{f},
#
# where :math:`\mathbf{u}\in\mathbb{R}^{2N}` and :math:`\mathbf{f}\in\mathbb{R}^{2N}`
# are 'flattened' versions of the displacement and forcing vectors. By solving this
# equation, we see how the structure of beams responds to the forcing.
#
# We begin by importing from the namespaces of Firedrake and Movement. ::

from firedrake import *
from movement import *

# Recall the initial uniform mesh of the unit square used in the Laplacian smoothing
# demo, which has four boundary segments tagged with the integers 1, 2, 3, and 4. Note
# that segment 4 corresponds to the top boundary. ::

import matplotlib.pyplot as plt
from firedrake.pyplot import triplot

n = 10
mesh = UnitSquareMesh(n, n)
coord_data_init = mesh.coordinates.dat.data.copy()
fig, axes = plt.subplots()
triplot(mesh, axes=axes)
axes.set_aspect(1)
axes.legend()
plt.savefig("lineal_spring-initial_mesh.jpg")

# .. figure:: lineal_spring-initial_mesh.jpg
#    :figwidth: 50%
#    :align: center
#
# We consider the same time-dependent forcing to the top boundary and see how the mesh
# structure responds. Recall the formula
#
# .. math::
#     \mathbf{f}(x,y,t)=\left[0, A\:\sin\left(\frac{2\pi t}T\right)\:\sin(\pi x)\right],
#
# where :math:`A` is the amplitude and :math:`T` is the time period. However, we cannot
# simply apply this formula in the same way as when solving the Poisson equation. We
# first need to apply a transformation to account for the fact that we are considering
# a discrete Poisson formulation. This amounts to applying the scale factor
#
# .. math::
#    K = \Delta x^2 / 4,
#
# where :math:`\Delta x` is the local mesh size. To account for this, we need to provide
# an additional index argument to :func:`forcing`. ::

import numpy as np

forcing_period = 1.0
num_timesteps = 10
timestep = forcing_period / num_timesteps
forcing_amplitude = 50


def forcing(index, x, t):
    K = mesh.cell_sizes.dat.data[index] ** 2 / 4
    return forcing_amplitude * np.sin(2 * pi * t / forcing_period) * np.sin(pi * x) * K


X = np.linspace(0, 1, n + 1)
times = np.arange(0, forcing_period + 0.5 * timestep, timestep)
boundary_nodes = DirichletBC(mesh.coordinates.function_space(), 0, 4).nodes

fig, axes = plt.subplots()
for t in times:
    axes.plot(X, forcing(boundary_nodes, X, t), label=f"t={t:.1f}")
axes.set_xlim([0, 1])
axes.legend()
box = axes.get_position()
axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axes.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("lineal_spring-forcings.jpg")

# .. figure:: lineal_spring-forcings.jpg
#    :figwidth: 60%
#    :align: center
#
# To apply this forcing, we need to create a :class:`~.SpringMover` instance and define
# a function for updating the forcing applied to the boundary nodes. ::

mover = SpringMover(mesh, method="lineal")


def update_forcings(t):
    coord_data = mover.mesh.coordinates.dat.data
    forcing_data = mover.f.dat.data
    for i in boundary_nodes:
        x, y = coord_data[i]
        forcing_data[i][1] = forcing(i, x, t)


# We are now able to apply the mesh movement method. The forcings effectively enforce a
# Dirichlet condition on the top boundary. On other boundaries, we enforce that there is
# no movement using the `fixed_boundaries` keyword argument. ::

import matplotlib.patches as mpatches

fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(12, 10))
for i, t in enumerate(times):
    idx = 0 if i == 0 else i + 1

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
plt.savefig("lineal_spring-adapted_meshes.jpg")

# .. figure:: lineal_spring-adapted_meshes.jpg
#    :figwidth: 100%
#    :align: center
#
# Again, the mesh is deformed according to the vertical forcing, with the left, right,
# and bottom boundaries remaining fixed, returning to be very close to its original state
# after one period. Let's check this in the :math:`\ell_\infty` norm. ::

coord_data = mover.mesh.coordinates.dat.data
linf_error = np.max(np.abs(coord_data - coord_data_init))
print(f"l_infinity error: {linf_error:.3f} m")
assert linf_error < 0.01

# Note that the mesh doesn't return to its original state quite as neatly with the lineal
# spring method as it does with the Laplacian smoothing method.
#
# We can view the sparsity pattern of the stiffness matrix as follows. ::

K = mover.stiffness_matrix
print(f"Stiffness matrix shape: {K.shape}")
print(f"Number of mesh vertices: {mesh.num_vertices()}")

fig, axes = plt.subplots()
axes.spy(K)
plt.savefig("lineal_spring-sparsity.jpg")

# .. figure:: lineal_spring-sparsity.jpg
#    :figwidth: 50%
#    :align: center
#
# This tutorial can be dowloaded as a `Python script <lineal_spring.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography:: demo_references.bib
#    :filter: docname in docnames
