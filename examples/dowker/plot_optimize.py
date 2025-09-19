
"""
.. _dowker_optimize_cycles_gallery:

Optimal cycles
=========================================

In this example we will

- generate a point cloud
- generate a collection of subsets (this can be viewed as a *cover*, and also as a *hypergraph*)
- compute the homology of the associated Dowker complex
- analyze cycle representatives
- plot cycle representatives
"""

# %%
import oat_python as oat

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.neighbors import NearestNeighbors
import itertools

# %% [markdown]
# Generate a point cloud
# -------------------------------------------------------

# %%
# Set parameter values
n_points            =   60
maxdis              =   None
maxdim              =   1

# %%
# Generate a point cloud consisting of three noisy circles
points               =   []
for seed in range(3):
    circle          =   oat.point_cloud.annulus(n_points=n_points, inner_radius=1, outer_radius=2.5, random_seed=seed)
    circle[:,0]     +=  2 * np.cos( seed * 2 * np.pi / 4)
    circle[:,1]     +=  2 * np.sin( seed * 2 * np.pi / 4)    
    points.append( circle )
points               =   np.concatenate(points)

dismat = sklearn.metrics.pairwise_distances(points)
dismat = ( dismat + dismat.T ) / 2

# %%
# Plot the point cloud
trace = go.Scatter(x=points[:,0],y=points[:,1], mode="markers")
fig = go.Figure([trace])
fig.update_layout( title=dict(text="Circle with noise"), height=650 )
fig

# %% [markdown]
# Choose a cover
# -------------------------------------------------------

# %%
radius_neighbor =   1.2; # hyperedges will be neighborhoods of vertices, with this radius
radius_net      =   1.0; # we'll make epsilon net with this value of epsilon

#   COMPUTE AN EPSILON NET
net, _          =   oat.dissimilarity.farthest_point_sampling( # the algorithm uses farthest point sampling
                        metric_space        =   dict( point_cloud = points ),
                        stopping_condition  =   dict( epsilon = radius_net )
                    ) 

#   PLOT THE COVER

data            =   []
data.append( go.Scatter(x=points[:,0],y=points[:,1], mode="markers", name="Cloud", showlegend=True)                                               )
data.append( go.Scatter(x=points[net,0],y=points[net,1], mode="markers", marker=dict(symbol="triangle-up", size=10), name="Net", showlegend=True) )


for counter, v in enumerate(net):
    point = points[v]
    trace = oat.plot.ball_2d( point[0], point[1], radius=radius_neighbor, n_points=100 )
    trace.update( opacity=0.2, name=f"Cover {counter}" )
    data.append(trace)

fig = go.Figure(data)
fig.update_layout( 
    title=dict(text="Hyperedges"), 
    height=650,
    yaxis=dict(
        scaleanchor="x",  # Anchor y-axis scaling to x-axis
        scaleratio=1.0    # Set the aspect ratio (1.0 for square aspect)
    ),    
)
fig

# %% [markdown]
# Compute homology
# -------------------------------------------------------
# 
# We'll compute the homology of
# - the dual hypergraph; that is, the hypergraph where vertices are balls, and for each vertex `v` we have a hyperedge that contains every ball to which `v` belongs
# - equivalently, the witness complex where every point is a witness and net points are landmarks
# 
# The homology solver only accepts hypergraphs represented by a list of lists of integers, currently.  If your hypergraph has a different format (e.g., if vertices are strings), then you can use some built-in tools to help translate back and forth between this format and list-of-list format; see the [rbs_reduced](rbs_reduced.ipynb) notebook for examples.

# %% [markdown]
# Format the cover as a family of sorted lists:

# %%

# data structure holding the whole cloud
net_wrapper         =   NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit( points[net] ) 
# witness complex where all points are witnesses, and net points are landmarks
cover               =   [ sorted(list(x)) for x in net_wrapper.radius_neighbors( points, radius=radius_neighbor, return_distance=False, ) ]    

# %% [markdown]
# Factor the boundary matrix to compute homology:

# %%
decomposition       =   oat.core.dowker.BoundaryMatrixDecompositionDowker( 
                            dowker_simplices            =   cover, 
                            max_homology_dimension      =   2
                        )

# %% [markdown]
# Inspect the Betti numbers

# %%
decomposition.betti_numbers()

# %% [markdown]
# Place a cycle basis for homology into a dataframe:

# %%
homology            =   decomposition.homology()
homology

# %% [markdown]
# Inspect a cycle representative

# %%
homology["cycle_representative"][1]

# %% [markdown]
# Plot cycles
# -------------------------------------------------------

# %%

#   DATA FOR THE POINT CLOUD

data                    =   []
data.append( go.Scatter(x=points[:,0],y=points[:,1], mode="markers", name="Cloud", showlegend=True)                                               )
data.append( go.Scatter(x=points[net,0],y=points[net,1], mode="markers+text", marker=dict(symbol="triangle-up", size=10), name="Net", showlegend=True, text=[str(x) for x in range(len(net))],  textposition='bottom center',  ) )

#   DATA FOR THE CYCLE

colors                  =   px.colors.qualitative.Plotly # specifies a (discrete) sequence of colors, represented by a list of strings

for rownum, row in homology.iterrows():
    if row["dimension"] != 1: continue

    trace_cycle         =   oat.plot.trace_2d_for_edges(
                                edges       =   row["cycle_representative"].simplex,
                                points      =   points[net],
                                name        =   f"Cycle {rownum}",
                            )
    data.append(trace_cycle) # add the trace to our list

#   PLOT

fig = go.Figure( data )
fig.update_layout(title="Coefficients still appear in the hover data", height=650 ) 
fig

# %% [markdown]
# Optimize
# -------------------------------------------------------

# %% [markdown]
# Within homology class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# This method will find a minimal cycle representative within the same homology class

# %%
optimal     =   decomposition.optimize_cycle(
                    unique_simplex_id=[7,15], 
                    problem_type="preserve homology class"
                )
optimal

# %% 
# Inspect the initial cycle
optimal["chain"]["initial_cycle"]


# %% 
# Inspect the optimal cycle
optimal["chain"]["optimal_cycle"]


# %% 
# Compare cycles visually

# %%

#   DATA FOR THE POINT CLOUD

data                    =   []
data.append( go.Scatter(x=points[:,0],y=points[:,1], mode="markers", name="Cloud", showlegend=True)                                               )
data.append( go.Scatter(x=points[net,0],y=points[net,1], mode="markers+text", marker=dict(symbol="triangle-up", size=10), name="Net", showlegend=True, text=[str(x) for x in range(len(net))],  textposition='bottom center',  ) )

#   DATA FOR THE CYCLE

colors                  =   px.colors.qualitative.Plotly # specifies a (discrete) sequence of colors, represented by a list of strings

for cycle_num, cycle_name in enumerate(["optimal_cycle", "initial_cycle"]):

    cycle_color         =   colors[ cycle_num % len(colors) ]

    for entrynum, entry in optimal["chain"][cycle_name].iterrows():
        edge            =   entry["simplex"]
        coefficient     =   entry["coefficient"]

        trace           =   oat.plot.trace_2d_for_edge( edge=edge, points=points[net] )
        trace.update( name=cycle_name, text=f"simplex {edge}<br>linear coefficent {coefficient}", opacity=0.7, line=dict(color=cycle_color,)) # customize appearance
        trace.update( legendgroup=cycle_num) # group edges that belong to the same cycle
        trace.update( showlegend = entrynum==0 )
        data.append(trace) # append to the data group

#   PLOT

fig = go.Figure( data )
fig.update_layout(title="Initial and optimal cycles", height=650 ) 
fig

# %% [markdown]
# Across homology classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# The following method allows us to optimize the cycle representative by adding not only boundaries, but cycle represntatives for other homology classes.

# %%
optimal     =   decomposition.optimize_cycle(
                    unique_simplex_id       =   (7,15),
                    problem_type            =   "preserve homology basis (once)"
                )
optimal



# %%
# Compare cycles visually

# %%

#   DATA FOR THE POINT CLOUD

data                    =   []
data.append( go.Scatter(x=points[:,0],y=points[:,1], mode="markers", name="Cloud", showlegend=True)                                               )
data.append( go.Scatter(x=points[net,0],y=points[net,1], mode="markers+text", marker=dict(symbol="triangle-up", size=10), name="Net", showlegend=True, text=[str(x) for x in range(len(net))],  textposition='bottom center',  ) )

#   DATA FOR THE CYCLE

colors                  =   px.colors.qualitative.Plotly # specifies a (discrete) sequence of colors, represented by a list of strings

for cycle_num, cycle_name in enumerate(["optimal_cycle", "initial_cycle"]):

    cycle_color         =   colors[ cycle_num % len(colors) ]

    for entrynum, entry in optimal["chain"][cycle_name].iterrows():
        edge            =   entry["simplex"]
        coefficient     =   entry["coefficient"]

        trace           =   oat.plot.trace_2d_for_edge( edge=edge, points=points[net] )
        trace.update( name=cycle_name, text=f"simplex {edge}<br>linear coefficent {coefficient}", opacity=0.7, line=dict(color=cycle_color,)) # customize appearance
        trace.update( legendgroup=cycle_num) # group edges that belong to the same cycle
        trace.update( showlegend = entrynum==0 )
        data.append(trace) # append to the data group

#   PLOT

fig = go.Figure( data )
fig.update_layout(title="Initial and optimal cycles", height=650 ) 
fig


