"""
.. _vietoris_submatrix_index_tool:

Submatrix Index Tool
========================================

A common pain point in computational homological algebra is passing between matrix and integer indices
for boundary and other matrices. OAT offers a :class:`oat_python.core.vietoris_rips.SubmatrixIndexTool`
to help wth this.
"""

# %% [markdown]
# 
# See also
# -------------------------------
# 
# See the :ref:`vietoris_rips_matrices` gallery for more examples!
# 
# 
# Setup
# -------------------------------

# %%
import oat_python as oat

from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.sparse import csr_matrix

# %%
# Generate a point cloud

points               =   oat.point_cloud.sphere_or_slice(n_points=100)

# %%
# Plot the point cloud
trace               =   go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2], mode="markers", marker=dict(opacity=1, size=4, color=points[:,2], colorscale="Aggrnyl"))
fig                 =   go.Figure(data=trace)
fig.update_layout(title=dict(text="Point cloud"), height=800,width=850) 
fig

# %% [markdown]
# Compute persistent homology
# -------------------------------

# %% [markdown]
# We compute persistent homology by factoring the boundary matrix.  The following cell generates a sparse distance matrix and feeds it to the persistent homology solver.  The result is a decomposition boundary matrix.  We will extract information from this matrix in the following cells.

# %%
# Prepare a dissimilarity matrix

dissimilarity_matrix    =   oat.dissimilarity.sparse_matrix_for_points(            
                                points                   =   points,
                                max_dissimilarity        =   0.5,
                            )

# %%
# Create and decompose the boundary matrix

decomposition           =   oat.core.vietoris_rips.BoundaryMatrixDecomposition(
                                dissimilarity_matrix    =   dissimilarity_matrix,
                                max_homology_dimension  =   1,
                            )

# %% [markdown]
# Access indices
# ------------------------------------
# 
# 

# %% [markdown]
# Create a `oat_python.core.vietoris_rips.SubmatrixIndexTool` whose rows and columns are the vertices and edges of the Vietoris-Rips complex, respectively.

# %%
submatrix_index_tool    =   decomposition.submatrix_index_tool(
                                row_dimensions      =   [0,],
                                column_dimensions   =   [1,]
                            )
submatrix_index_tool

# %% [markdown]
# The number of row and column indices can be accessed via lookup functions:

# %%
submatrix_index_tool.number_of_rows()

# %%
submatrix_index_tool.number_of_columns()

# %% [markdown]
# The (ordered) lists of row and column indices can also be exported to dataframe format, for inspection:

# %%
oat.plot.display_dataframes_side_by_side(
    submatrix_index_tool.row_indices()[:5],
    submatrix_index_tool.column_indices()[:5],
    titles=["Row indices (vertices)","Column indices (edges)"]
)

# %% [markdown]
# Write a submatrix to CSR
# ----------------------------------
# 
# A common task in data analysis is to export a submatrix of the boundary matrix into standard Python sparse matrix format, such as SciPy's CSR. In order to do this, you have to specify which rows and columns you want, and in what order. The :class:`oat_python.core.vietoris_rips.SubmatrixIndexTool` lets you specify these criteria exactly.

# %%
boundary_matrix         =   decomposition.boundary_matrix_oracle()
boundary_submatrix      =   boundary_matrix.write_submatrix_to_csr(
                                submatrix_index_tool
                            )
boundary_submatrix

# %%
plt.figure(figsize=(6, 6))
plt.spy(boundary_submatrix, markersize=2, color="orange")
plt.title(r'Boundary Matrix ($\partial_1$)', y=-0.18)  # Move title to the bottom
plt.tight_layout()

# %% [markdown]
# Change index types
# ----------------------------------------------------------------

# %% [markdown]
# Get a row vector from the CSR matrix

# Right now we have to convert row coefficients to fractions, but in the future this will be taken care of automatically
row_vector             =   boundary_submatrix[0].toarray().flatten()
row_vector             =   np.vectorize(Fraction.from_float)(row_vector) 
row_vector[:10]

# %% [markdown]
# Convert integer indices to simplex indices:

# %%
chain                   =   submatrix_index_tool.row_vector_dataframe_for_dense_array(
                                row_vector
                            )
chain


