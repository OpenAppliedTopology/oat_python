"""
Plotting tools.

This module contains plotting tools for simplicial complexes, surfaces, barcodes, persistence diagrams, etc.
"""


import oat_python
import oat_python.barcode




import copy
import itertools
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import re

from cmath import inf
from sklearn import manifold
from IPython.display import display, HTML
from . import hypergraph




#   ========================================
#   DATAFRAMES
#   ========================================







def display_dataframes_side_by_side(*dfs, titles=()):
    """
    Display multiple pandas DataFrames side by side in a Jupyter notebook.

    Parameters
    ----------
    *dfs : pandas.DataFrame
        One or more DataFrames to display side by side.
    titles : iterable of str, optional 
        Titles for each DataFrame. If provided, must match the number of DataFrames.

    Returns
    -------
    None
        Displays the DataFrames side by side in the notebook.

    Example
    ------- 
    .. code-block:: python

        import pandas as pd
        from oat_python.plot import display_side_by_side

        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4, 5]})
        df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8, 9]})
        display_side_by_side(df1, df2, titles=['First DataFrame', 'Second DataFrame'])
    """
    html_str = ''
    for i, df in enumerate(dfs):
        title = f'<h4>{titles[i]}</h4>' if i < len(titles) else ''
        html_str += f'<div style="display:inline-block; vertical-align:top; margin-right:20px;">{title}{df.to_html()}</div>'
    html = HTML(html_str)
    return html



#   ========================================
#   PERSISTENCE DIAGRAM + BARCODE
#   ========================================






def persistence_diagram_guidelines( upper_limit ):
    """
    Generate Plotly guideline traces for a persistence diagram.

    Given a scalar value ``upper_limit``, this function returns two Plotly Scatter traces:
    - A solid diagonal line segment from (0, 0) to (upper_limit, upper_limit).
    - A dashed horizontal guideline from (0, upper_limit) to (upper_limit, upper_limit).

    Parameters
    ----------
    upper_limit : float
        The upper bound for the guideline lines in the persistence diagram.

    Returns
    -------
    traced : plotly.graph_objs.Scatter
        Scatter trace for the diagonal line segment (0, 0) -- (upper_limit, upper_limit), solid.
    traceh : plotly.graph_objs.Scatter
        Scatter trace for the horizontal line segment (0, upper_limit) -- (upper_limit, upper_limit), dashed.

    Example
    -------
    .. code-block:: python

        import plotly.graph_objects as go
        from oat_python.plot import persistence_diagram_guidelines

        traced, traceh = persistence_diagram_guidelines(1.0)
        fig = go.Figure([traced, traceh])
        fig
    """
    traceh = go.Scatter(x=[0,upper_limit], y=[upper_limit,upper_limit], mode="lines")
    traceh.update( line=dict( dash="dot", color="black" ) )
    traced = go.Scatter(x=[0,upper_limit],y=[0,upper_limit], mode="lines", )
    traced.update( line=dict( color="black" ) )    
    return traced, traceh

def persistence_diagram( persistent_homology_dataframe, guideline_limit = None ):
    """
    Create a Plotly figure for a persistence diagram.

    Parameters
    ----------
    persistent_homology_dataframe : pandas.DataFrame
        DataFrame with numeric columns labeled ``birth``, ``death``, and ``dimension``.
        Optionally, may include ``birth simplex``, ``num_cycle_simplices``, and ``num_bounding_simplices`` columns for richer hovertext.
    guideline_limit : float, optional
        If provided, sets the upper limit for the diagram guidelines.
        The diagonal guideline is drawn at 1.2 × guideline_limit, and the horizontal guideline at 1.1 × guideline_limit.
        If not provided, the function determines suitable limits from the data.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A Plotly figure displaying the persistence diagram.

    Notes
    -----
    - Each point represents a persistent feature, colored by its homology dimension.
    - Hovering over a point shows detailed information, including birth/death filtration, interval length,
      birth simplex, and (if present) cycle and bounding chain sparsity.
    - Infinite death values are replaced by a finite proxy for plotting.
    - The function adds diagonal and horizontal guideline lines for visual reference.

    Example
    -------
    .. code-block:: python

        import pandas as pd
        from oat_python.plot import persistence_diagram

        persistent_homology_dataframe = pd.DataFrame({
            "birth_filtration": [0.1, 0.2],
            "death_filtration": [0.5, float("inf")],
            "dimension": [0, 1],
            "birth_simplex": [(0,), (1,)],
        })
        fig     =   persistence_diagram(
                        persistent_homology_dataframe
                    )
        fig
    """
    fig                         =   go.Figure( data=[] )


    from cmath import inf  
    C                               =   guideline_limit
    if C is None:
        finite_endpoints = { float(x) for x in persistent_homology_dataframe['birth_filtration'].tolist() + persistent_homology_dataframe['death_filtration'].tolist()  if x != inf }
        if len(finite_endpoints)==0:
            C                       =   1
            infinity_proxy          =   1 # determines whwere we will draw the "infinity" line
            diagonal_limit          =   1.1 # how much farther than "infinity proxy" 
        else:
            fin_max                 =   max( finite_endpoints )
            fin_min                 =   min( finite_endpoints )
                        
            C                       =   max( finite_endpoints )
            if fin_max > fin_min:
                infinity_proxy      =   C + 0.1 * (fin_max - fin_min)
                diagonal_limit      =   C + 0.2 * (fin_max - fin_min)
            else:
                infinity_proxy      =   C + 1.1 * np.abs(C)
                diagonal_limit      =   C + 0.2 * np.abs(C)

    


    fig.add_hline(y= infinity_proxy, line=dict(dash="dot"))    
    trace_lined, trace_lineh    =   persistence_diagram_guidelines( diagonal_limit ) 
    trace_lined.update(showlegend=False)#name="x=y")
    fig.add_trace(trace_lined)    

    

    # fig = px.scatter()
    colors = px.colors.qualitative.Plotly;

    
    flagged_dimensions  =   set()

    for index, row in persistent_homology_dataframe.iterrows():
        x               =   row['birth_filtration']
        y               =   np.minimum( row['death_filtration'], infinity_proxy )
        dimension       =   row['dimension']
        color           =   colors[ dimension ]
        text            =   f"birth filtration = {x}<br>" + \
                            f"death filtration = {y}<br>" +\
                            f"interval length = { row['death_filtration'] - row['birth_filtration']}<br>" +\
                            f"birth simplex = {row['birth_simplex']}<br>"
        
        if 'num_cycle_simplices' in row.keys():
            text += f"cyle representative num simplices = {row['num_cycle_simplices']}<br>"
        if 'num_bounding_simplices' in row.keys():
            text += f"bounding chain num simplices = {row['num_bounding_simplices']}<br>"

        text            +=  f"row of homology dataframe = {index}"
        
        trace   =   go.Scatter(
                        x                   =   [x],
                        y                   =   [y],
                        mode                =   "markers",
                        text                =   [text],
                        showlegend          =   dimension not in flagged_dimensions,
                        legendgroup         =   dimension,
                        marker              =   dict(color=color),
                        name                =   f"Dimension {dimension}",
                    )
        fig.add_trace(trace)

        flagged_dimensions.add(dimension)


    fig.update_layout(yaxis_range=[-0.1 * C, 1.2 * C ].sort())    
    fig.update_layout(xaxis_range=[-0.1 * C, 1.2 * C ].sort())    
    fig.update_layout(height=600,width=650)    
    return fig



def barcode( persistent_homology_dataframe, guideline_limit = None ):
    """
    Create a Plotly figure for the barcode of a filtered chain complex.

    Parameters
    ----------
    persistent_homology_dataframe : pandas.DataFrame
        DataFrame with numeric columns labeled ``birth``, ``death``, and ``dimension``.
        Optionally, may include ``birth simplex``, ``death simplex``, ``num_cycle_simplices``, and ``num_bounding_simplices`` columns for richer hovertext.
    guideline_limit : float, optional
        If provided, sets the upper limit for the vertical guideline (drawn at 1.1 × guideline_limit).
        If not provided, the function determines a suitable limit from the data.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A Plotly figure displaying the barcode diagram.

    Notes
    -----
    - Each horizontal bar represents a persistent feature, colored by its homology dimension.
    - Hovering over a bar shows detailed information, including birth/death filtration, birth/death simplex,
      cycle and bounding chain sparsity, and the row index in the DataFrame.
    - Infinite death values are replaced by a finite proxy for plotting.
    - The function adds a vertical guideline for visual reference.

    Example
    -------
    .. code-block:: python

        import pandas as pd
        from oat_python.plot import barcode

        persistent_homology_dataframe = pd.DataFrame({
            "birth_filtration": [0.1, 0.2, 0.3],
            "death_filtration": [0.5, float("inf"), 0.8],
            "dimension": [0, 1, 2],
            "birth_simplex": [(0,), (1,2), (6,7,8)],
            "death_simplex": [(1,), (3,4,5), (9,10,11)],
            "num_cycle_simplices": [3, 4, 7],
            "num_bounding_simplices": [5, 6, 8],
        })
        fig     =   barcode(
                        persistent_homology_dataframe
                    )
        fig
    """
    fig                         =   go.Figure( data=[] )

    # fig.add_trace( go.Scatter(x=[],y=[],yaxis="y2") )   
    C                           =   guideline_limit
    if C is None:
        C                       =   oat_python.barcode.max_finite_value( persistent_homology_dataframe['birth_filtration'].tolist() + persistent_homology_dataframe['death_filtration'].tolist() )
        if C is None:
            C                   =   1

    fig.add_vline(x= 1.1* C, line=dict(dash="dot"))    


    intervals                   =   sorted(
                                        list(
                                            zip(
                                                persistent_homology_dataframe['dimension'],
                                                persistent_homology_dataframe['birth_filtration'], 
                                                persistent_homology_dataframe['death_filtration'], 
                                                persistent_homology_dataframe.index 
                                            )
                                        )
                                    )    
    
    color_sequence              =   px.colors.qualitative.Plotly
    num_colors                  =   len(color_sequence)
    max_dim                     =   max(persistent_homology_dataframe['dimension'])
    x                           =   [[] for _ in range(max_dim + 1)]
    y                           =   [[] for _ in range(max_dim + 1)]
    hovertext                   =   [[] for _ in range(max_dim + 1)]

    # coordinates and hover text for bars
    for bar_counter, (dim,birth,death,id) in enumerate(intervals):
        x[dim].append( birth )
        x[dim].append( np.minimum( death, 1.1 * C ) )
        x[dim].append( np.nan )
        y[dim].append( bar_counter + 1 )
        y[dim].append( bar_counter + 1 )
        y[dim].append( np.nan )    
        newtext                 = \
        f"birth filtration {birth}<br>" \
        + f"death filtration {death}<br>" \
        + f"birth simplex {persistent_homology_dataframe['birth_simplex'][id]}<br>" \
        + f"death simplex  {persistent_homology_dataframe['death_simplex'][id]}<br>"
        if 'num_cycle_simplices' in persistent_homology_dataframe.keys():
            newtext              += f"cycle representative num simplices {persistent_homology_dataframe['num_cycle_simplices'][id]}<br>"
        if 'num_bounding_simplices' in persistent_homology_dataframe.keys():
            newtext              += f"bounding chain num simplices {persistent_homology_dataframe['num_bounding_simplices'][id]}<br>"
        newtext += f"row of data frame {id}"
        hovertext[dim].append( newtext )         

    for dimension, (x,y,hovertext) in enumerate(zip(x,y,hovertext)):
        # traces for bars
        color                   =   color_sequence[ dimension % num_colors ]        
        trace                   =   go.Scatter( x=x, y=y, name=f"Dim {dimension}", mode='lines', line=dict(color=color), )
        hov                     =   [ t for t in hovertext for _ in range(3) ]
        trace.update(hoverinfo="text")
        trace.update(hovertext=hov)
        fig.add_trace(trace) 

        # # traces for hover text
        # trace                   =   go.Scatter( x=x[::3], y=y[::3], mode='markers+text', marker=dict(color=color), legendgroup=dimension, )        
        # trace.update(hoverinfo="text")
        # trace.update(name=f"Dim {dimension}")
        # trace.update(hovertext=hovertext)

        # add trace to figure
        # fig.add_trace(trace)     

    fig.update_layout(yaxis_range=[ 0, bar_counter + 2 ])    
    fig.update_layout(xaxis_range=[-0.1 * C, 1.2 * C ])    
    fig.update_layout(height=600,width=600)    
    return fig



#   =======================
#   SIMPLICES
#   =======================


def fig_3d_for_simplices( 
        simplices, 
        points, 
        kwargs_points=dict(),
        kwargs_edges=dict(),
        kwargs_triangles=dict(), 
        kwargs_layout=dict()
    ):
    """
    Plots a collection of simplices in 3D.

    Parameters
    ----------
    simplices : iterable of sequence of int, shape (n_simplices, k)
        An iterable (e.g., list or array) where each element is a sequence of integers,
        representing the vertices of a simplex.
    points : indexable
        An object that maps integer vertex labels to 3D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 3).
    kwargs_points : dict, optional
        Additional keyword arguments to pass to the Plot for vertices.
        These arguments are passed to the `plotly.graph_objects.Scatter3d <https://plotly.com/python/reference/scatter3d/>`_ constructor.
        The only keywords which are set by default are

        - ``mode = 'markers'``
        - ``marker = dict(size=6, color="crimson")``
        - ``text = [ str(point_index) for point_index in points.keys() ]`` if ``points`` is a dictionary
          and ``text = [ str(p) for p in range(len(points)) ]`` if ``points`` is a list or array
        - ``name = "Vertices"``
    kwargs_edges : dict, optional
        Additional keyword arguments to pass to the Plot for edges.
        These arguments are passed to the `plotly.graph_objects.Scatter3d <https://plotly.com/python/reference/scatter3d/>`_ constructor.
        The only keywords which are set by default are 
        
        - ``mode="lines"``
        - ``line = dict(color="crimson", width=4)``
        - ``name = "Edges"``

    kwargs_triangles : dict, optional
        Additional keyword arguments to pass to the Plot for triangles.
        These arguments are passed to the `plotly.graph_objects.Mesh3d <https://plotly.com/python/reference/mesh3d/>`_ constructor.
        The only keywords which are set by default are

        - ``opacity = 0.2``
        - ``showlegend = True``   
        - ``name = "Triangles"``

    kwargs_layout : dict (default: None)
        Additional keyword arguments to pass to the Plot layout.
        - If None is provided, then background color is set to white, and gridlines, tick labels, and axis titles are removed.
        - If an empty dictionary is provided, then the default Plotly layout is used.
        - **NOTE** The layout can be modified post-hoc for customization. Some examples include:

          - ``fig.update_layout(template="plotly_dark")`` for a dark theme with light-colored legend text and grid lines
          - ``oat_python.set_background_color(fig, background_color="navy")`` for a navy background with light-colored legend text and grid lines

        **NOTE** If you'd like a white or black background without gridlines,
        use ``oat_python.plot.set_background_color(fig, color="white")`` or ``oat_python.plot.set_background_color(fig, color="black")``.
         

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A Plotly figure for the the simplices
        
        - This figure contains a list of three traces, which can be accessed via ``fig.data``.
          The list has form [trace_vertices, trace_edges, trace_triangles]:

          - `trace_vertices` is a `plotly.graph_objects.Scatter3d <https://plotly.com/python/reference/scatter3d/>`_ trace for the vertices.
          - `trace_edges` is a `plotly.graph_objects.Scatter3d <https://plotly.com/python/reference/scatter3d/>`_ trace for the edges.
          - `trace_triangles` is a `plotly.graph_objects.Mesh3d <https://plotly.com/python/reference/mesh3d/>`_ trace for the triangles.   
        
        - The layout of the figure can be accessed via ``fig.layout``.
    
    Example
    -------
    See the :ref:`styling_3d_gallery` gallery for examples and styling.
    """

    # reformat points as a dictionary if needed
    if not isinstance(points, dict):
        points              =   { i: point for i, point in enumerate(points) }

    for point in points.values():
        if len(point) < 3:
            raise ValueError("One or more of the points passed to fig_3d_for_simplices has fewer than 3 coordinates.")

    
    
    # extract a list of all triangles contained in the provided list of simplices (including faces of higher-dimensional simplices)
    triangles           =   oat_python.simplex.dimension_m_faces_for_simplices(
                                simplices,
                                m = 2,
                            )
    
    # generate a trace for the triangles
    if not "opacity" in kwargs_triangles:
        kwargs_triangles['opacity'] = 0.3
    if not "showlegend" in kwargs_triangles:
        kwargs_triangles['showlegend'] = True
    if not "color" in kwargs_triangles:
        kwargs_triangles['color'] = "orange"
    if not "name" in kwargs_triangles:
        kwargs_triangles['name'] = "Triangles"
    trace_triangles     =   oat_python.plot.trace_3d_for_triangles(
                                triangles             =   triangles,
                                points                =   points,
                                **kwargs_triangles,
                            )

    # extract a list of all triangles contained in the provided list of simplices (including faces of higher-dimensional simplices)
    edges               =   oat_python.simplex.dimension_m_faces_for_simplices(
                                simplices,
                                m = 1,
                            )
    
    # trace for edges
    if not "mode" in kwargs_edges:
        kwargs_edges['mode'] = "lines"
    if not "line" in kwargs_edges:
        kwargs_edges['line'] = dict(color="white", width=4)
    if not "name" in kwargs_edges:
        kwargs_edges['name'] = "Edges"
    trace_edges         =   oat_python.plot.trace_3d_for_edges(
                                edges                 =   edges,
                                points                =   points,
                                **kwargs_edges,
                            )
    
    # trace for the vertices
    if not "mode" in kwargs_points:
        kwargs_points['mode'] = "markers"
    if not "text" in kwargs_points:
        kwargs_points['text'] = [ str(point_index) for point_index in points.keys() ]
    if not "marker" in kwargs_points:
        kwargs_points['marker'] = dict(size=5, color="white")
    if not "name" in kwargs_points:
        kwargs_points['name'] = "Vertices"
    trace_vertices      =   go.Scatter3d(
                                x                   =   [ point[0] for point in points.values() ],
                                y                   =   [ point[1] for point in points.values() ],
                                z                   =   [ point[2] for point in points.values() ],
                                **kwargs_points,
                            )    
    
    fig = go.Figure( data=[ trace_vertices, trace_edges, trace_triangles ] )
    
    fig.update_layout(
        **kwargs_layout
    )
    return fig



#   =======================
#   CYCLE REPRESENTATIVES
#   =======================


def contrast_initial_and_optimal_cycles_in_3d( 
        boundary_matrix_decomposition, 
        birth_simplex, 
        points, 
        kwargs_initial=dict(), 
        kwargs_optimal=dict(), 
        kwargs_surface=dict(),
        kwargs_points=dict(),
        kwargs_layout=dict(),
    ):
    """
    Generates a Plotly figure comparing an initial cycle representative with an optimal cycle representative.

    Parameters
    ----------
    boundary_matrix_decomposition : :class:`oat_python.core.vietoris_rips.BoundaryMatrixDecomposition`

    birth_simplex : list or tuple of int
        The birth simplex of the persistent homology class whose cycle representative is to be compared.

    points : indexable
        An object that maps integer vertex labels to 3D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 3).

    kwargs_initial : dict, optional
        Additional keyword arguments to pass to the Plot for the initial cycle.
        These arguments are passed to the `plotly.graph_objects.Scatter3d <https://plotly.com/python/reference/scatter3d/>`_ constructor.
        The only keywords which are set by default are
        
        - ``line="white"``
        - ``name="Initial cycle"``

    kwargs_optimal : dict, optional
        Additional keyword arguments to pass to the Plot for the optimal cycle.
        These arguments are passed to the `plotly.graph_objects.Scatter3d <https://plotly.com/python/reference/scatter3d/>`_ constructor.
        The only keywords which are set by default are 
        
        - ``line="crimson"``
        - ``name="Optimal cycle"``

    kwargs_surface : dict, optional
        Additional keyword arguments to pass to the Plot for the surface between the cycles.
        These arguments are passed to the `plotly.graph_objects.Mesh3d <https://plotly.com/python/reference/mesh3d/>`_ constructor.
        The only keywords which are set by default are

        - ``opacity=0.5``
        - ``color="white"``
        - ``name="Surface between cycles"``.

    kwargs_points : dict, optional
        Additional keyword arguments to pass to the Plot for the point cloud.
        These arguments are passed to the `plotly.graph_objects.Scatter3d <https://plotly.com/python/reference/scatter3d/>`_ constructor.
        The  keywords which are set by default are as follows (points are slightly transparent and colored by their y-coordinate):
        
        - ``mode="markers"``
        - ``marker=dict(opacity=0.8, size=3, color=[point[1] for point in points], colorscale="Peach")``
        - ``opacity=0.8``
        - ``name="Point cloud"``.

    kwargs_layout : dict, optional
        Additional keyword arguments to pass to the Plot layout.
        If None is provided, then background color is set to black, and gridlines, tick labels, and axis titles are removed.
        If an empty dictionary is provided, then the default Plotly layout is used.
        The keywords which are set by default are
        
        - ``title=dict(text="A cycle before and after optimization")``
        - ``template="plotly_dark"``
        - ``height=1000``
        - ``width=1200``.  

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A Plotly figure comparing the initial and optimal cycle representatives, along with the surface between them
        and the underlying point cloud.

    Example
    -------

    See :ref:`vietoris_rips_dragon` for an example.

    Notes
    -----

    - To view the source code for this funciton, click the "[source]" link to the right of the function signature at the top of this docstring.

    """

    optimal_cycle_data      =   boundary_matrix_decomposition.optimize_cycle(
                                    birth_simplex                   =   birth_simplex, 
                                    problem_type                    =   "preserve PH basis",
                                    verbose                         =   False,
                                )    
    
    intial_cycle            =   boundary_matrix_decomposition \
                                    .change_of_basis_matrix_oracle() \
                                    .column_for_simplex(
                                        birth_simplex 
                                    )

    edges_initial           =   intial_cycle['simplex'].tolist()
    edges_optimal           =   optimal_cycle_data['chain']['optimal_cycle']['simplex'].tolist()
    triangles               =   optimal_cycle_data['chain']['surface_between_cycles']['simplex'].tolist() # the chain that bounds the difference between the cycles
    

    if not "line" in kwargs_initial:
        kwargs_initial['line'] = dict(color="white", width=10)
    if not "name" in kwargs_initial:
        kwargs_initial['name'] = "Initial cycle"

    if not "line" in kwargs_optimal:
        kwargs_optimal['line'] = dict(color="crimson", width=10)
    if not "name" in kwargs_optimal:
        kwargs_optimal['name'] = "Optimal cycle"   


    if not "showlegend" in kwargs_surface:
        kwargs_surface['showlegend'] = True
    if not "opacity" in kwargs_surface:
        kwargs_surface['opacity'] = 0.5
    if not "color" in kwargs_surface:
        kwargs_surface['color'] = "white"
    if not "name" in kwargs_surface:
        kwargs_surface['name'] = "Surface between cycles"

    if not "mode" in kwargs_points:
        kwargs_points['mode'] = "markers"
    if not "marker" in kwargs_points:
        kwargs_points['marker'] = dict(opacity=0.8, size=3, color=[point[1] for point in points], colorscale="Peach")
    if not "opacity" in kwargs_points:
        kwargs_points['opacity'] = 0.8
    if not "name" in kwargs_points:
        kwargs_points['name'] = "Point cloud"

    trace_edges_initial =   oat_python.plot.trace_3d_for_edges(
                                edges=edges_initial,
                                points=points,
                                **kwargs_initial,                     
                            )
    trace_edges_optimal =   oat_python.plot.trace_3d_for_edges(
                                edges=edges_optimal,
                                points=points,
                                **kwargs_optimal,                           
                            )
    trace_triangles     =   oat_python.plot.trace_3d_for_triangles(
                                triangles=triangles,
                                points=points, 
                                **kwargs_surface,                           
                            ) 
    trace_points         =   go.Scatter3d(
                                x=points[:,0],
                                y=points[:,1],
                                z=points[:,2], 
                                **kwargs_points,
                            )

    fig                 =   go.Figure(
                                data=[
                                    trace_edges_initial, 
                                    trace_edges_optimal, 
                                    trace_triangles, 
                                    trace_points,
                                ]
                            )
    
    if not "title" in kwargs_layout:
        kwargs_layout['title'] = dict(text="A cycle before and after optimization")
    if not "template" in kwargs_layout:
        kwargs_layout['template'] = "plotly_dark"
    if not "height" in kwargs_layout:
        kwargs_layout['height'] = 1000
    if not "width" in kwargs_layout:
        kwargs_layout['width'] = 1200

    fig.update_layout( **kwargs_layout )

    return fig



#   =======================
#   MDS FOR PLOTTING
#   =======================


#   HELPER FUNCTIONS TO GENERATE MDS HOP COORDINATES
#   -------------------------------------------------





def vertex_embedding_for_networkx_graph(
        G, 
        dimension=2, 
        method="spring",
        **kwargs 
    ):
    """
    Computes an embedding of the nodes of a NetworkX graph.

    Parameters
    ----------
    G : networkx.Graph
        A NetworkX graph whose nodes will be embedded.
    dimension : int, optional
        The dimension of the embedding space (e.g., 2 for planar, 3 for 3D). Default is 2.
    method : str, (optional, default: "spring")
        The method used to generate the embedding. The options are:

        - "spring": Use a spring layout to position nodes before computing hop distances.
        - "hop_mds": Uses multidimensional scaling (MDS) applied to the hop distance matrix. The hop distance between nodes is the shortest path length in the graph.
    **kwargs
        Additional keyword argumentsto
         
        - if ``method = "spring"`` these arguments pass to the `networkx.spring_layout` function. Examples include
        
          - ``iterations`` the maximum number of iterations to perform (default: 50).
          - ``seed`` for reproducibility.
          - `complete list: `networkx.spring_layout documentation <https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html>`_
        
        - if ``method = "hop_mds"`` these arguments pass to the `sklearn.manifold.MDS` constructor. Examples include
        
          - ``max_iter`` the maximum number of iterations to perform (default: 300).
          - ``random_state`` for reproducibility.
          - `complete list: `sklearn.manifold.MDS documentation <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html>`_

    Returns
    -------
    coords : dict
        A dictionary mapping each vertex label to its coordinates in the embedding (as a NumPy array of length `dimension`).

    Example
    -------
    .. code-block:: python

        import networkx as nx
        from oat_python.plot import vertex_embedding_for_networkx_graph
        import plotly.graph_objects as go

        G = nx.path_graph([0, 1, 2, 3])
        coords = vertex_embedding_for_networkx_graph(G, dimension=2)
        print(coords)

        # 2D Plotly scatter plot of the embedding
        x = [coords[v][0] for v in G.nodes]
        y = [coords[v][1] for v in G.nodes]
        fig = go.Figure(go.Scatter(x=x, y=y, mode="markers+text", text=list(G.nodes)))
        fig.update_layout(title="2D MDS Embedding of Graph Nodes")
        fig
    """

    if method == "spring":
        pos = nx.spring_layout(
            G, 
            dim=dimension,
            **kwargs
        )
        return { k: np.array(pos[k]) for k in pos.keys() }
    
    elif method == "hop_mds":
        D, vertex_labels    = oat_python.dissimilarity.hop_distance_for_networkx_graph(G)

        mds = manifold.MDS(
            n_components    =   dimension,
            dissimilarity   =   "precomputed",
            **kwargs,
        )
        coords = mds.fit(D).embedding_    

        return { vertex_labels[k]: coords[k] for k in range(len(vertex_labels)) }
    else:
        raise ValueError(f"Unknown value for keword argument 'method': {method}. Supported methods are 'spring' and 'hop_mds'.")



def vertex_embedding_for_simplices(simplices, dimension=2, method="hop_mds", **kwargs):
    """
    Computes an embedding of the vertices of a collection of simplices.

    Parameters
    ----------
    simplices : list of list of int
        A list of simplices, where each simplex is represented as a list of vertex indices.
    dimension : int, optional
        The dimension of the embedding space (e.g., 2 for planar, 3 for 3D). Default is 2.
    method : str (optional, default: "spring")
        The method used to generate the embedding. The options are:

        - "spring": Use a spring layout to position nodes before computing hop distances.
        - "hop_mds": Uses multidimensional scaling (MDS) applied to the hop distance matrix. The hop distance between nodes is the shortest path length in the graph.
    **kwargs
        Additional keyword argumentsto
         
        - if ``method = "spring"`` these arguments pass to the `networkx.spring_layout` function. Examples include
        
          - ``iterations`` the maximum number of iterations to perform (default: 50).
          - ``seed`` for reproducibility.
          - `complete list: `networkx.spring_layout documentation <https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html>_`
        
        - if ``method = "hop_mds"`` these arguments pass to the `sklearn.manifold.MDS` constructor. Examples include
        
          - ``max_iter`` the maximum number of iterations to perform (default: 300).
          - ``random_state`` for reproducibility.
          - `complete list: sklearn.manifold.MDS documentation <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html>_`

    Returns
    -------
    coords : dict
        A dictionary mapping each vertex label to its coordinates in the embedding (as a NumPy array of length `dimension`).    
    """
    G = oat_python.simplex.networkx_graph_for_simplices(simplices)
    return  vertex_embedding_for_networkx_graph(
                G,
                dimension=dimension, 
                method=method, 
                **kwargs
            )


#   ========================================
#   Vertices
#   ========================================




def trace_2d_for_vertices( points, **kwargs ):
    """
    Generate a 2D Plotly Scatter (``plotly.graph_objects.Scatter``) trace for a collection of vertices.

    Parameters
    ----------
    points : indexable
        An object that maps integer vertex labels to 2D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 2).
    **kwargs
        Additional keyword arguments to pass to the Plotly Scatter constructor. See `Plotly documentation <https://plotly.com/python/reference/scatter/>`__ for details.
        The only keyword which is set by default is ``mode="markers"``.

    Returns
    -------
    trace : plotly.graph_objs.Scatter
        A Plotly Scatter trace for the vertices.

    Notes
    -----
    This function provides minimal processing, and is provided simply for convenience.
    Check out the source code by clicking the "[source]" link to the right of the function signature at the top of this docstring.
    """
    
    # Accepts points as either a sequence (list/array) or a dict[int] -> vector
    if isinstance(points, dict):
        x = [point[0] for point in points.values()]
        y = [point[1] for point in points.values()]
    else:
        x = [point[0] for point in points]
        y = [point[1] for point in points]

    if not 'mode' in kwargs.keys():
        kwargs['mode'] = "markers"

    return go.Scatter(x=x,y=y, **kwargs)







def trace_3d_for_vertices( points, **kwargs ):
    """
    Generate a 3D Plotly Scatter3d trace (``plotly.graph_objects.Scatter3D``) for a collection of vertices.

    Parameters
    ----------
    points : indexable
        An object that maps integer vertex labels to 3D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 3).
    **kwargs
        Additional keyword arguments to pass to the Plotly Scatter3d constructor. See `Plotly documentation <https://plotly.com/python/reference/scatter3d/>`__ for details.
        The only keyword which is set by default is ``mode="markers"``.

    Returns
    -------
    trace : plotly.graph_objs.Scatter3d
        A Plotly Scatter3d trace for the vertices.

    Notes
    -----
    This function provides minimal processing, and is provided simply for convenience.
    Check out the source code by clicking the "[source]" link to the right of the function signature at the top of this docstring.        
    """

    if isinstance(points, dict):
        points = [point for point in points.values()]

    for point in points:
        if len(point) < 3:
            raise ValueError("One or more of the points passed to trace_3d_for_vertices has fewer than 3 coordinates.")
    
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    if not 'mode' in kwargs.keys():
        kwargs['mode'] = "markers"

    return go.Scatter3d(x=x,y=y,z=z, **kwargs)











#   ========================================
#   EDGES
#   ========================================



def trace_2d_for_edge( edge, points, **kwargs ):
    """
    Generate a 2D Plotly Scatter (``plotly.graph_objects.Scatter``) trace for a single edge.

    Parameters
    ----------
    edge : sequence of int, length 2
        A sequence (e.g., list, tuple, array) of two integers, representing the vertices of the edge.
    points : indexable
        An object that maps integer vertex labels to 2D or 3D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 2) or (n_vertices, 3).
    **kwargs
        Additional keyword arguments to pass to the Plotly Scatter constructor. See `Plotly documentation <https://plotly.com/python/reference/scatter/>`__ for details.
        The only keyword which is set by default is ``mode="lines"``.

    Returns
    -------
    trace : plotly.graph_objs.Scatter
        A Plotly Scatter trace for the edge, with an "extra" vertex in the center (which is invisible but
        allows the user to view data when the cursor passes over the center point; see below for an example).

    Notes
    -----
    - For fine-grained control, edges are returned one-by-one rather than as a single trace.
    - You can simulate grouped toggling in Plotly by setting the ``legendgroup`` property on each trace.
    - To change the color of the plot, use ``trace.update(line_color="black")`` or another color.
    - To add custom hover information, set the ``text`` field of the returned trace and set ``hoverinfo="text"`` or ``hovertemplate`` as desired.

    Examples
    --------
    See :ref:`edges_2d_gallery`.
    """    

    x0, x1  =   points[edge[0]][0], points[edge[1]][0]
    y0, y1  =   points[edge[0]][1], points[edge[1]][1]
    x       =   [ x0, (x0+x1)/2, x1 ]
    y       =   [ y0, (y0+y1)/2, y1 ]
    if not 'mode' in kwargs.keys():
        kwargs['mode'] = "lines"
    return go.Scatter(x=x, y=y, **kwargs)




def trace_2d_for_edges( edges, points, **kwargs ):
    """
    Generate a 2D scatter trace (`plotly.graph_objects.Scatter <https://plotly.com/python/reference/scatter/>`_) for a collection of edges.

    Parameters
    ----------
    edges : iterable of sequence of int, shape (n_edges, 2)
        An iterable (e.g., list or array) where each element is a sequence of two integers,
        representing the vertices of an edge.
    points : indexable
        An object that maps integer vertex labels to 2D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 2).
    **kwargs
        Additional keyword arguments to pass to the Plotly Scatter constructor. See `Plotly documentation <https://plotly.com/python/reference/scatter/>`__ for details.
        The only keyword which is set by default is ``mode="lines"``.

    Returns
    -------
    trace : plotly.graph_objs.Scatter
        A Plotly Scatter trace containing all edges.

    Notes
    -----
    - Values passed in ``**kwargs`` can be updated later using ``trace.update(...)``.
    - This method is slightly less flexible than repeated calls to ``trace_2d_for_edge``, which allows for greater
      customization, but is kept for convenience and as an example of plotting multiple disjoint edges in a single trace.

    Examples
    --------

    See :ref:`edges_2d_gallery`.

    Notes
    -----

    The trace returned by this function can be updated later using ``trace.update(...)``.
    In order to perform this update it is helpful to know how the trace is constructed.
    In fact, :func:`trace_2d_for_edges` performs only minimal processing:

    - Clicke the "[source]" link to the right of the function name, above, to view the source code.
    - A copy is also provided below for convenience, but this copy may **become outdated if the function is updated in the future**, so refer to the source link as the authoritative resource.
    - Note that multiple disjoint line segments are created by separating pairs of endpoint coordinates (x0,y0),(x1,y1) values with ``None``.

    .. code-block:: python

        def trace_2d_for_edges( edges, points, **kwargs ):
            x       =   []
            y       =   []
            for edge in edges:
                x = x + [ points[edge[0]][0], points[edge[1]][0], None ]
                y = y + [ points[edge[0]][1], points[edge[1]][1], None ] 
            if not 'mode' in kwargs.keys():
                kwargs['mode'] = "lines"
            return go.Scatter(x=x,y=y, **kwargs)        
    """
    x       =   []
    y       =   []
    for edge in edges:
        x = x + [ points[edge[0]][0], points[edge[1]][0], None ]
        y = y + [ points[edge[0]][1], points[edge[1]][1], None ] 
    if not 'mode' in kwargs.keys():
        kwargs['mode'] = "lines"
    return go.Scatter(x=x,y=y, **kwargs)



def trace_3d_for_edge( edge, points, **kwargs ):
    """
    Generate a 3D Plotly Scatter3d trace (``plotly.graph_objects.Scatter3D``) for a single edge.

    Parameters
    ----------
    edge : sequence of int, length 2
        A sequence (e.g., list, tuple, array) of two integers, representing the vertices of the edge.
    points : indexable
        An object that maps integer vertex labels to 3D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 3).
    **kwargs
        Additional keyword arguments to pass to the Plotly Scatter3d constructor. See `Plotly documentation <https://plotly.com/python/reference/scatter3d/>`__ for details.
        The only keyword which is set by default is ``mode="lines"``.

    Returns
    -------
    trace : plotly.graph_objs.Scatter3d
        A Plotly Scatter3d trace for the edge, with an "extra" vertex in the center (which is invisible but
        allows the user to view data when the cursor passes over the center point; see
        below for an example).    

    Examples
    --------
    
    See :ref:`edges_3d_gallery`.

    Notes
    -----

    The trace returned by this function can be updated later using ``trace.update(...)``.
    In order to perform this update it is helpful to know how the trace is constructed.
    In fact, :func:`trace_3d_for_edge` performs only minimal processing:
    
    - Check out the full source code by clicking the "[source]" link to the right of the function name, above
    - A copy is also provided below for convenience, but this copy may **become outdated if the function is updated in the future**, so refer to the source link as the authoritative resource.
    - Notice that an extra vertex is added at the midpoint of the edge, which allows the user to hover over the center of the edge and see hover text (if provided).      

    .. code-block:: python
    
        def trace_3d_for_edge( edge, points, **kwargs ):
            x0, x1  =   points[edge[0]][0], points[edge[1]][0]
            y0, y1  =   points[edge[0]][1], points[edge[1]][1]
            z0, z1  =   points[edge[0]][2], points[edge[1]][2]    
            x       =   [ x0, (x0+x1)/2, x1 ]
            y       =   [ y0, (y0+y1)/2, y1 ]
            z       =   [ z0, (z0+z1)/2, z1 ]     
            if not 'mode' in kwargs.keys():
                kwargs['mode'] = "lines"    
            return go.Scatter3d(x=x,y=y,z=z, **kwargs)

    """

    if len(points[edge[0]]) < 3 or len(points[edge[1]]) < 3:
        raise ValueError("One or more of the points passed to trace_3d_for_edge has fewer than 3 coordinates.")

    x0, x1  =   points[edge[0]][0], points[edge[1]][0]
    y0, y1  =   points[edge[0]][1], points[edge[1]][1]
    z0, z1  =   points[edge[0]][2], points[edge[1]][2]    
    x       =   [ x0, (x0+x1)/2, x1 ]
    y       =   [ y0, (y0+y1)/2, y1 ]
    z       =   [ z0, (z0+z1)/2, z1 ]     
    if not 'mode' in kwargs.keys():
        kwargs['mode'] = "lines"    
    return go.Scatter3d(x=x,y=y,z=z, **kwargs)



def trace_3d_for_edges( edges, points, **kwargs ):
    """
    Generate a 3D scatter trace (`plotly.graph_objects.Scatter3D <https://plotly.com/python/reference/scatter3d/>`_) for a collection of edges.

    Parameters
    ----------
    edges : iterable of sequence of int, shape (n_edges, 2)
        An iterable (e.g., list or array) where each element is a sequence of two integers,
        representing the vertices of an edge.
    points : indexable
        An object that maps integer vertex labels to 3D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 3).
    **kwargs
        Additional keyword arguments to pass to the Plotly Scatter3d constructor. See `Plotly documentation <https://plotly.com/python/reference/scatter3d/>`__ for details.
        The only keyword which is set by default is ``mode="lines"``.

    Returns
    -------
    trace : plotly.graph_objs.Scatter3d
        A Plotly Scatter3d trace containing all edges.

    Notes
    -----
    Individual edges cannot be assigned separate colors in a single trace.
    If different colors are desired, create separate traces for each edge.
    See :ref:`edges_2d_gallery` for an example.

    The trace returned by this function can be updated later using ``trace.update(...)``.
    In order to perform this update it is helpful to know how the trace is constructed.
    In fact, :func:`trace_2d_for_edges` performs only minimal processing.

    - Check out the full source code by clicking the "[source]" link to the right of the function name, above.
    - A copy is also provided below for convenience, but this copy may **become outdated if the function is updated in the future**, so refer to the source link as the authoritative resource.
    - Note that multiple disjoint line segments are created by separating pairs of endpoint coordinates (x0,y0,z0),(x1,y1,z1) values with ``None``.

    .. code-block:: python

        def trace_3d_for_edges( edges, points, **kwargs ):
            x       =   []
            y       =   []
            z       =   []
            for edge in edges:
                x = x + [ points[edge[0]][0], points[edge[1]][0], None ]
                y = y + [ points[edge[0]][1], points[edge[1]][1], None ] 
                z = z + [ points[edge[0]][2], points[edge[1]][2], None ]  
            if not 'mode' in kwargs.keys():
                kwargs['mode'] = "lines"              
            return go.Scatter3d(x=x,y=y,z=z, **kwargs)            

    Examples
    --------

    See :ref:`edges_3d_gallery`.
    """

    for edge in edges:
        if len(points[edge[0]]) < 3 or len(points[edge[1]]) < 3:
            raise ValueError("One or more of the points passed to trace_3d_for_edges has fewer than 3 coordinates.")


    x       =   []
    y       =   []
    z       =   []
    for edge in edges:
        x = x + [ points[edge[0]][0], points[edge[1]][0], None ]
        y = y + [ points[edge[0]][1], points[edge[1]][1], None ] 
        z = z + [ points[edge[0]][2], points[edge[1]][2], None ]  
    if not 'mode' in kwargs.keys():
        kwargs['mode'] = "lines"              
    return go.Scatter3d(x=x,y=y,z=z, **kwargs)




#   ============================
#   TRIANGLES
#   ============================



def trace_3d_for_triangles(triangles=[], points=[], **kwargs):
    """
    Generate a 3D mesh trace (`plotly.graph_objects.Mesh3D <https://plotly.com/python/reference/mesh3d/>`_) for a collection of triangles in 3D.

    Parameters
    ----------
    triangles : array-like of shape (n_triangles, 3)
        An array or list of lists of integers, where each inner list contains three indices
        referring to the vertices of a triangle.
    points : indexable object
        An indexable object (such as a dictionary or array) that maps vertex indices to x-y-z coordinates,
        e.g., a dictionary of tuples or a NumPy array of shape (n_vertices, 3).
    **kwargs
        Additional keyword arguments to pass to the Plotly Mesh3d constructor. 
        
        - See `Plotly documentation <https://plotly.com/python/reference/mesh3d/>`__ for details.
        - **Caution**: The keyword arguments ``vertexcolor``, ``intensity``, ``text``, and ``hovertext`` are handled specially, see below for details.

    Returns
    -------
    trace : plotly.graph_objs.Mesh3d
        A single Plotly Mesh3d trace containing all triangles.

    Examples
    -------
    
    See :ref:`triangles_3d_gallery`.

    Assigning values to vertices
    ------------------------------------

    TL;DR
    ^^^^^^^^^^^^^^^^^
    
    If you want to set values for the ``vertexcolor``, ``intensity``, ``text``, or ``hovertext`` keyword arguments
    in a plot created by this function, there are two different options:

    1. Pass these arguments directly to the function, via the ``**kwargs`` argument. In this case
       there are no special requirements, simply pass an indexable object (e.g., list, array, or dictionary)
       with values for all vertices that appear in the ``triangles`` argument. *Aside: this functionality is actually
       a bit more flexible than the normal Plotly Mesh3d constructor, because you can pass values for only the vertices
       that appear in the triangles, rather than for all vertices from 0 to N.*
    2. Update the returned trace, e.g., ``trace.update(vertexcolor=...)``. In this case you can only pass
         lists or arrays, e.g. ``[color0, color1, ..., colorN]`` where ``colori`` is the color assigned to the `i`th
         largest vertex value. For example, if ``triangles = [(3,5,11), (5,7,8)]``,
         then
         
         - ``color0`` is the color assigned to vertex ``3``
         - ``color1`` is the color assigned to vertex ``5`` 
         - ``color2`` is the color assigned to vertex ``7``
         - ``color3`` is the color assigned to vertex ``8`` 
         - ``color4`` is the color assigned to vertex ``11``

         To obtain the list of vertex labels in sorted order, you can use one of the following methods:

            - ``vertex_labels = sorted(list(set().union(*triangles)))``
            - ``vertex_labels = oat_python.simplex.vertices_incident_to_simplices(triangles)``
            - (after creating the trace) ``vertex_labels = trace.meta['vertex_labels']``

    Technical background
    ^^^^^^^^^^^^^^^^^^^^^^^

    Plotly allows you to assign values to vertices for color and hover text
    by passing lists, e.g. ``[color0, color1, ...]``.
    In order to use these arguments, you must know which vertex corresponds to each color or size value in the list.
    That is, you need a list of vertex labels ``[v0, v1, ..., vN]``. 

    In the normal usage of Plotly Mesh3d plots this is not a problem. The Mesh3d constructor takes ``x``, ``y``, and ``z`` arguments,
    which are lists of the x, y, and z coordinates of each vertex, respectively, together with ``i``, ``j``, and ``k`` arguments,
    which are lists of the vertex indices for each triangle. For example,
    if ``i = [0, 1, 2]``, ``j = [1, 2, 0]``, and ``k = [2, 0, 1]``, then the triangles are
    ``(v0, v1, v2)``, ``(v1, v2, v0)``, and ``(v2, v0, v1)``. The coordinates of vertex ``vi`` are ``(x[i], y[i], z[i])``. So the list
    of vertex labels is simply 0, 1, 2, ..., N.

    This is the only format in which Plotly Mesh3d plots can be created, so there's no functionality to pass a single
    triangle ``(3,5,7)`` together with x-y-z coordinates for just the vertices ``3``, ``5``, and ``7``. In order to work
    around this limitation, the :func:`oat_python.plot.trace_3d_for_triangles` function reassigns vertex labels as needed,
    so that the set of all vertex labels forms a contiguous sequence ``0, 1, 2, ... N``. It then permutes the ``x``, ``y``, ``z``
    values appropriately, and discards any x-y-z coordinates that are not needed. (*Discarding unused coordinates is
    one of the main motivations for using this function.*)

    So, if you ever want to modify a trace produced by :func:`oat_python.plot.trace_3d_for_triangles`,
    you need to know the original vertex label for each of the vertices labeled ``0, 1, 2, ... N`` in the trace.
    There are several ways to obtain this list:

    - place the set of all vertex labels into a list and sort it, e.g., ``vertex_labels = sorted(list(set().union(*triangles)))``.
    - call ``oat_python.simplex.vertices_incident_to_simplices(triangles)``
    - (after creating the trace) look up ``trace.meta['vertex_labels']``

    Any one of these methods will give you a list ``[v0, v1, ..., vN]``, where ``vi`` is the original label of the vertex
    that was relabeled as ``i``.  

    If you pass values for ``vertexcolor``, ``intensity``, ``text``, or ``hovertext`` to
    :func:`oat_python.plot.trace_3d_for_triangles` via the ``**kwargs`` argument, then the function will take care of
    relabeling the values appropriately, so you can simply pass an indexable object (e.g., list, array, or dictionary)
    with values for all vertices that appear in the ``triangles`` argument. Otherwise use
    the list ``[v0, v1, ..., vN]`` to assign values to vertices in the trace after it has been created.
    """

    # Plotly Mesh3d requires x, y, z to be formatted as lists, tuples, or arrays
    # for this reason, to avoid bundling a large point cloud into this trace, we
    # relabel the vertices incident to the simplices with labels {1, ..., n}, and
    # then generate coordinate array with only as many coordinates as we need

    triangles = np.array(triangles)

    # nvl2ovl = new vertex label to old vertex label
    # ovl2nvl = old vertex label to new vertex label
    # triangles_relabeled = triangles with new vertex labels
    nvl2ovl = oat_python.simplex.vertices_incident_to_simplices(triangles)
    nvl2ovl = [int(p) for p in nvl2ovl]
    ovl2nvl = { nvl2ovl[nvl]: nvl for nvl in range(len(nvl2ovl))}
    triangles_relabeled = [ [ ovl2nvl[x] for x in simplex] for simplex in triangles ]
    triangles_relabeled = np.array(triangles_relabeled)


    # ensure points have 3 coordinates each
    for v in nvl2ovl:
        if len(points[v]) < 3:
            raise ValueError("One or more points of the points passed to trace_3d_for_triangles has fewer than 3 coordinates.")

    # place the x-y-z coordinates into an array
    coordinates_as_rows     =   np.array( 
                                    [ points[ v ][:3] for v in nvl2ovl ] 
                                )
    
    # in the edge case of no vertices, we must reshape the array to ensure it has shape (0,3)
    if coordinates_as_rows.shape[0] == 0:
        coordinates_as_rows =   coordinates_as_rows.reshape( 0, 3 ) 
    
    if "vertexcolor" in kwargs:
        kwargs['vertexcolor'] = [ kwargs['vertexcolor'][v] for v in nvl2ovl ]    
    if "intensity" in kwargs:
        if not ( "intensitymode" in kwargs and kwargs['intensitymode'] == "cell" ):
            kwargs['intensity'] = [ kwargs['intensity'][v] for v in nvl2ovl ]
    if "text" in kwargs:
        if not isinstance(kwargs['text'], str): # if text is a single string, don't try to index into it
            kwargs['text'] = [ kwargs['text'][v] for v in nvl2ovl ]
    if "hovertext" in kwargs:
        if not isinstance(kwargs['hovertext'], str): # if hovertext is a single string, don't try to index into it
            kwargs['hovertext'] = [ kwargs['hovertext'][v] for v in nvl2ovl ]
    

    trace   =   go.Mesh3d(
                    x=coordinates_as_rows[:,0],
                    y=coordinates_as_rows[:,1],
                    z=coordinates_as_rows[:,2],
                    i = [x[0] for x in triangles_relabeled],
                    j = [x[1] for x in triangles_relabeled],
                    k = [x[2] for x in triangles_relabeled],
                    meta = { "vertex_labels": nvl2ovl },
                    **kwargs,
                )    
    return trace 


def trace_2d_for_triangle( triangle=[], points=[], **kwargs ):
    """
    Generates a filled Plotly trace for a triangle in 2D.

    Parameters
    ----------
    triangle : iterable of int, length 3
        Indices of the triangle's vertices.
    points : indexable
        An object that maps integer vertex labels to 2D or 3D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 2) or (n_vertices, 3).
    **kwargs
        Additional keyword arguments to pass to the Plotly Scatter constructor. See `Plotly documentation <https://plotly.com/python/reference/scatter/>`__ for details.
        The only keywords which are set by default are ``fill="toself"`` and ``mode="none"``.

    Returns
    -------
    trace : plotly.graph_objs.Scatter
        A Plotly Scatter trace representing the filled triangle.

    Notes
    -----
    - Values passed in ``**kwargs`` can be updated later using ``trace.update(...)``.

    Examples
    --------

    See :ref:`triangles_2d_gallery`.
    """
    x = []; y = []; 
    for p in range(3):
        vertex = triangle[ p ]
        x.append( points[vertex][0] )
        y.append( points[vertex][1] )
    if not 'fill' in kwargs.keys():
        kwargs['fill'] = "toself"
    if not 'mode' in kwargs.keys():
        kwargs['mode'] = "none"
    return go.Scatter(x=x, y=y, **kwargs)

def trace_2d_for_triangles(
        triangles=[], 
        points=[], 
        single_trace=True, 
        **kwargs
    ):
    """
    Generates Plotly traces for a collection of triangles in 2D.

    Parameters
    ----------
    triangles : iterable of iterable of int, shape (n_triangles, 3)
        An iterable (e.g., list or array) where each element is a sequence of three integers,
        representing the indices of the triangle's vertices.
    points : indexable
        An object that maps integer vertex labels to 2D coordinates, e.g., a dictionary of tuples,
        a list of coordinate tuples, or a NumPy array of shape (n_vertices, 2).
    single_trace : bool, optional
        If True, returns a single Plotly Scatter trace containing all triangles (separated by None values).
        If False, returns a list of individual Scatter traces, one per triangle. Default is True.
    **kwargs
        Additional keyword arguments to pass to the Plotly Scatter constructor.
        See `Plotly documentation <https://plotly.com/python/reference/scatter/>`__ for details.
        The only keyword which is set by default is ``fill="toself"``.

    Returns
    -------
    trace or list of traces
        If `single_trace` is True, returns a single trace. 
        If False, returns a list of such traces, one for each triangle.
        Each trace has type Plotly Scatter with ``fill="toself"``.

    Notes
    -----
    - Values passed in ``**kwargs`` can be updated later using ``trace.update(...)``.

    .. warning::

        Plotly may yield unexpected behavior when triangles overlap. **In particular, in overlapping regions the color fill may disappear.**
        This can be observed in the code example below.    

    Examples
    --------

    See :ref:`triangles_2d_gallery`.
    """

    if not 'fill' in kwargs.keys():
        kwargs['fill'] = "toself"

    # Plotly Mesh3d requires x, y, z to be formatted as lists, tuples, or arrays
    # for this reason, to avoid bundling a large point cloud into this trace, we
    # relabel the vertices incident to the simplices with labels {1, ..., n}, and
    # then generate coordinate array with only as many coordinates as we need
    if single_trace:
        x = []
        y = []
        for triangle in triangles:    
            # generate a closed loop for each triangle    
            for p in range(4):
                vertex = triangle[ p % 3 ]
                x.append( points[vertex][0] )
                y.append( points[vertex][1] )
            # mark the break between triangles with None
            x.append(None)
            y.append(None)
        
        trace = go.Scatter(x=x, y=y, **kwargs) 
        # fig = go.Figure(go.Scatter(x=[0,1,2,0,None,3,3,5,5,3], y=[0,2,0,0,None,0.5,1.5,1.5,0.5,0.5], fill="toself"))    
        return trace 
    else:
        data = []
        for triangle in triangles:    
            x = []
            y = []
            # generate a closed loop for each triangle    
            for p in range(4):
                vertex = triangle[ p % 3 ]
                x.append( points[vertex][0] )
                y.append( points[vertex][1] )
            # mark the break between triangles with None
            x.append(None)
            y.append(None)
            data.append( go.Scatter(x=x, y=y, **kwargs)  )
        return data         








#   ==============================================================================
#   SHAPES 2D
#   ==============================================================================


def ball_2d( x, y, radius, n_points ):
    """
    Returns a trace for a ball of radius ``radius`` centered at ``(x,y)``.  The perimiter
    of the ball is a closed piecewise linear curve with ``n_points`` vertices.
    """
    theta = np.linspace( 0, 2 * np.pi, n_points )    
    return go.Scatter(
        x   =   x + radius * np.cos(theta), 
        y   =   y + radius * np.sin(theta), 
        fill="toself"
    )




#   ==============================================================================
#   WIRE DIAGRAM 3D
#   ==============================================================================


def wire_sphere_3d(x, y, z, radius, nlattitude, nlongitude):
    """
    Plot a wire mesh sphere.

    Parameters
    ----------
    x : float
        x coordinate of the center of the sphere.
    y : float
        y coordinate of the center of the sphere.
    z : float
        z coordinate of the center of the sphere.
    radius : float
        Radius of the sphere.
    nlattitude : int
        Number of latitude lines (horizontal rings) to draw on the sphere.
    nlongitude : int
        Number of longitude lines (vertical segments) to draw on the sphere.

    Returns
    -------
    data : list of plotly.graph_objs.Scatter3d
        A list of Plotly Scatter3d traces. Each trace represents a latitude or longitude line,
        and together they form a wireframe mesh of the sphere. Pass this list to ``go.Figure(data)``
        to visualize the wireframe sphere in 3D.

    Notes
    -----
    - You can cause all these traces to toggle on/off together by calling ``trace.update(legendgroup="your_legend_group_name")`` on each trace;
      see the [Plotly legend documentation](https://plotly.com/python/legend/) for details and examples.
    - You can remove all but one of the traces from the legend by calling ``trace.update(showlegend=False)`` on the traces you wish to hide.

    Example
    -------
    .. code-block:: python

        import plotly.graph_objects as go
        from oat_python.plot import wire_sphere_3d

        traces = wire_sphere_3d(x=0, y=0, z=0, radius=1, nlattitude=10, nlongitude=20)
        for trace in traces:
            trace.update(legendgroup="sphere", showlegend=False)
        traces[0].update(showlegend=True, name="Wire Sphere")
        fig = go.Figure(traces)
        fig
    """

    nlongitude = nlongitude + 1 # this corrects for the fact that we overlap 1 line

    # Define the phi and theta values for the sphere
    phi = np.linspace(0, np.pi, nlattitude)
    theta = np.linspace(0, 2*np.pi, nlongitude)

    # Compute the x, y, and z coordinates of the sphere points
    x = x + radius * np.outer(np.sin(phi), np.cos(theta)).ravel()
    y = y + radius * np.outer(np.sin(phi), np.sin(theta)).ravel()
    z = z + radius * np.outer(np.cos(phi), np.ones_like(theta)).ravel()

    # Define the indices of the points on each latitudinal line
    indices = [np.arange(i*nlongitude, (i+1)*nlongitude) for i in range(nlattitude)]

    # Define the indices of the points on each longitudinal line
    for i in range(nlongitude):
        indices.append(np.arange(i, nlattitude*nlongitude, nlongitude))

    #   Create the longitudinal lines
    data    =   [go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='white', width=1))]

    # Add the latitudinal lines to the plot
    for index in indices:
        data.append(go.Scatter3d(x=x[index], y=y[index], z=z[index], mode='lines', line=dict(color='black', width=1)))    

    return data



def wire_rectangle(x0,x1,y0,y1,z0,z1,):
    """
    The edges (1-cells) of a 3d rectangle of form ``[x0,x1] x [y0,y1] x [z0,z1]``
    
    :return data: a list of Plotly Scatter3d traces, in ``lines`` mode
    """
    data = []
    for x in [x0,x1]:
        for y in [y0,y1]:
            trace   =   go.Scatter3d(x=[x,x], y=[y,y], z=[z0,z1], mode="lines")
            data.append(trace)
    for x in [x0,x1]:
        for z in [z0,z1]:
            trace   =   go.Scatter3d(x=[x,x], y=[y0,y1], z=[z,z], mode="lines")
            data.append(trace)
    for z in [z0,z1]:
        for y in [y0,y1]:
            trace   =   go.Scatter3d(x=[x0,x1], y=[y,y], z=[z,z], mode="lines")
            data.append(trace)                        
    return data

#   ==============================================================================
#   SURFACE PLOTS
#   ==============================================================================



#   CUBE
#   --------------------------------------
def surface_cube(x0,y0,z0, width=1, anchor="center"):
    """
    Generate a Plotly Surface trace for a 3-dimensional cube.

    The cube has side length equal to ``width`` and is positioned according to the anchor point.

    Parameters
    ----------
    x0 : float
        x coordinate of the anchor point.
    y0 : float
        y coordinate of the anchor point.
    z0 : float
        z coordinate of the anchor point.
    width : float, optional
        Side length of the cube. Default is 1.
    anchor : {'left', 'center'}, optional
        If ``anchor='left'``, the anchor point is the vertex with minimal x, y, and z coordinates.
        If ``anchor='center'``, the anchor point is the center of the cube. Default is 'center'.

    Returns
    -------
    trace : plotly.graph_objs.Surface
        A Plotly Surface trace for the cube.
    x : numpy.ndarray
        The x coordinates of the cube mesh.
    y : numpy.ndarray
        The y coordinates of the cube mesh.
    z : numpy.ndarray
        The z coordinates of the cube mesh.

    Notes
    -----
    This is a convenience wrapper around the ``surface_rectangle`` function.
    The cube is defined as ``[x0, x0+width] x [y0, y0+width] x [z0, z0+width]`` if ``anchor='left'``,
    or centered at (x0, y0, z0) if ``anchor='center'``.

    Example
    -------
    .. code-block:: python

        trace, x, y, z = surface_cube(0, 0, 0, width=2, anchor="center")
    """
    # strange errors occur (due to rounding) if we don't convert everything to float
    if anchor == "left":    
        x1 = float(x0) + width
        y1 = float(y0) + width
        z1 = float(z0) + width
    elif anchor == "center":
        x0 = float(x0) - width/2
        y0 = float(y0) - width/2
        z0 = float(z0) - width/2
        x1 = float(x0) + width
        y1 = float(y0) + width
        z1 = float(z0) + width
    else:
        raise ValueError('The "anchor" keyword argument must be "left" or "center"')
        
    return surface_rectangle(x0,x1,y0,y1,z0,z1)


#   RECTANGLE
#   --------------------------------------
def surface_rectangle(x0,x1,y0,y1,z0,z1):
    """
    A Plotly Surface trace for a 3-dimensional rectangle of form [x0,x1] x [y0,y1] x [z0,z1].

    Returns one trace and three coordinate matrices: ``go.Surface(z=z, x=x, y=y), x, y, z``.
    The coordinates tend to be useful for adjusting surface color.

    **Remark** This tends to produce cleaner results than a Plotly mesh plot.

    **How it works**

    The visual intuition is to image a cloth 3 units wide and 4 units long being
    folded over the surface of a cube.  The lefthand 3x3 units of cloth cover the top
    5 faces of the cube, and the righthand 1x3 units tuck under to cover the base.
    Some squares collapse down to lines. For reference, calling 

    .. code-block:: python

        _, x, y, z = rectangle_trace(0,1,2,3,4,5)
        print(x,"\\n\\n",y,"\\n\\n",z)

    will return

    .. code-block::

        [[0 0 1 1 0]
        [0 0 1 1 0]
        [0 0 1 1 0]
        [0 0 1 1 0]] 

        [[2 2 2 2 2]
        [2 2 2 2 2]
        [3 3 3 3 3]
        [3 3 3 3 3]] 

        [[4 4 4 4 4]
        [4 5 5 4 4]
        [4 5 5 4 4]
        [4 4 4 4 4]]
    """
    # strange errors occur (due to rounding) if we don't convert everything to float
        
    z = np.full((4,5), float(z0))
    z[1:3,1:3] = float(z1)

    x = np.full((4,5), float(x0))
    x[:,2:4] = float(x1)

    y = np.full((4,5), float(y0))
    y[2:,:]=float(y1)

    return go.Surface(z=z, x=x, y=y), x, y, z,


#   SPHERE
#   --------------------------------------
def surface_sphere(x, y, z, radius, resolution=20):
    """
    Generate a Plotly Surface trace for a sphere.

    Parameters
    ----------
    x : float
        x coordinate of the center of the sphere.
    y : float
        y coordinate of the center of the sphere.
    z : float
        z coordinate of the center of the sphere.
    radius : float
        Radius of the sphere.
    resolution : int, optional
        Number of points in the discretization of the sphere; the higher the resolution, the smoother the sphere will look. Default is 20.

    Returns
    -------
    trace : plotly.graph_objs.Surface
        A Plotly Surface trace for the sphere.
    x : numpy.ndarray
        The x coordinates of the sphere mesh.
    y : numpy.ndarray
        The y coordinates of the sphere mesh.
    z : numpy.ndarray
        The z coordinates of the sphere mesh.

    Notes
    -----
    The sphere is discretized using spherical coordinates. The ``u`` parameter
    represents the azimuthal angle (longitude) and the ``v`` parameter represents
    the polar angle (latitude). The ``x``, ``y``, and ``z`` coordinates are then
    computed using the formulas:

    .. code-block:: python

        x = radius * np.cos(u) * np.sin(v) + x
        y = radius * np.sin(u) * np.sin(v) + y
        z = radius * np.cos(v) + z

    The resulting coordinates are then used to create a Plotly Surface trace.
    """
    x = float(x)     # strange errors occur (due to rounding) if we don't convert everything to float
    y = float(y)
    z = float(z)
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    x = radius * np.cos(u)*np.sin(v) + x
    y = radius * np.sin(u)*np.sin(v) + y
    z = radius * np.cos(v) + z
    trace = go.Surface(x=x,y=y,z=z), x, y, z
    return trace

    


#   ==============================================================================
#   MESH PLOTS
#   ==============================================================================



#   RECTANGLE TRACE (MESH - WORSE RESULTS)
#   --------------------------------------
def mesh_rectangle(x0,x1,y0,y1,z0,z1):
    """
    Generate a Plotly Mesh3d trace for a rectangle of form ``[x0, x1] x [y0, y1] x [z0, z1]``.

    Parameters
    ----------
    x0 : float
        x coordinate of the left edge.
    x1 : float
        x coordinate of the right edge.
    y0 : float
        y coordinate of the bottom edge.
    y1 : float
        y coordinate of the top edge.
    z0 : float
        z coordinate of the near edge.
    z1 : float
        z coordinate of the far edge.

    Returns
    -------
    trace : plotly.graph_objs.Mesh3d
        A Plotly Mesh3d trace for the rectangle.

    Notes
    -----
    This method is under review for deprecation, as it tends to produce poor results (e.g., poor shadowing and seams at non-right angles).
    Consider using ``surface_rectangle`` as an alternative.
    """
    print("This method tends to generate poor shadowing and seams at non-right angles.  Consider ``rectangle_trace_3d`` as an alternative.")
    
    # strange errors occur (due to rounding) if we don't convert everything to float
    x0=float(x0); x1=float(x1); y0=float(y0); y1=float(y1); z0=float(z0); z1=float(z1);

    x = np.array([0, 0, 1, 1, 0, 0, 1, 1]) * (x1-x0) + x0
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0]) * (y1-y0) + y0
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * (z1-z0) + z0   

    return go.Mesh3d(
        # 8 vertices of a cube
        x=x,
        y=y,
        z=z,
        colorbar_title='z',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [1, 'magenta']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity = np.linspace(0, 1, 8, endpoint=True),
        # i, j and k give the vertices of triangles
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        name='y',
        showscale=True
    ) 



def surface_octahedron():
    """
    A Plotly Surface trace for an octahedron.
    The octahedron is a polyhedron with eight triangular faces, twelve edges, and six vertices.

    The octahedron is defined by the following vertices:
    - (1, 0, 0)
    - (-1, 0, 0)
    - (0, 1, 0)
    - (0, -1, 0)
    - (0, 0, 1)
    - (0, 0, -1)
    """
    points = np.array( # coordinate oracle
        [
            # first four columns "walk around the equator along adjacent vertices"
            # the final two columns represent the north/south poles
            [ -1,  0, 1,  0,  0,  0 ], # x
            [  0, -1, 0,  1,  0,  0 ], # y
            [  0,  0, 0,  0, -1,  1 ], # z
        ]
    ).T

    edges = [ [0,1], [1,2], [2,3], [3,0] ]
    triangles = [ edge + [pole] for edge in edges for pole in [4,5] ]

    trace       =   oat_python.plot.trace_3d_for_triangles( 
                        triangles, 
                        points      =   points,
                        showlegend  =   True,
                        color       =   "red",
                        opacity     =   0.5,
                    )
    return trace















#   ==============================================================================
#   MODIFY BACKGROUND
#   ==============================================================================


def contains_3d_trace(fig):
    """
    Returns ``True`` if the Plotly figure contains one or more 3D traces.

    Parameters
    ----------
    fig : plotly.graph_objs.Figure
        A Plotly Figure object.

    Returns
    -------
    bool
        True if the figure contains at least one 3D trace, False otherwise.

    Notes
    -----
    This function works by checking whether any of the traces in the figure
    is an instance of one of the following types:
        
        - plotly.graph_objects.Scatter3d
        - plotly.graph_objects.Surface
        - plotly.graph_objects.Mesh3d
        - plotly.graph_objects.Volume
        - plotly.graph_objects.Isosurface
        - plotly.graph_objects.Cone
        - plotly.graph_objects.Streamtube
        - plotly.graph_objects.Volume

    Example
    -------
    .. code-block:: python

        import plotly.graph_objects as go
        from oat_python.plot import contains_3d_trace

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=[1,2], y=[3,4], z=[5,6]))
        print(contains_3d_trace(fig))  # Output: True

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[1,2], y=[3,4]))

        print(contains_3d_trace(fig2))  # Output: False
    """
    plot_3d_trace_types = [
        plotly.graph_objects.Scatter3d,
        plotly.graph_objects.Surface,
        plotly.graph_objects.Mesh3d,
        plotly.graph_objects.Volume,
        plotly.graph_objects.Isosurface,
        plotly.graph_objects.Cone,
        plotly.graph_objects.Streamtube,
        plotly.graph_objects.Volume,
    ]

    for trace in fig.data:
        for type in plot_3d_trace_types:
            if isinstance(trace, type):
                return True
    return False


def blank_background(fig):
    """
    Remove grid lines, zero lines, tick labels, and axis titles from a 3D Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objs.Figure
        A Plotly Figure object containing a 3D scatter or surface plot.

    Returns
    -------
    None
        The function modifies the input figure in place.

    Example
    -------
    See the :ref:`styling_3d_gallery` gallery for examples and styling.
    """
    # If the figure has a 3D scene, modify it one way
    if contains_3d_trace(fig):
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showgrid=False,  # Hide x-axis grid lines
                    showticklabels=False, # Hide x-axis tick labels
                    zeroline=False, # Hide the x=0 plane
                    title="" # Remove x-axis title
                ),
                yaxis=dict(
                    showgrid=False,  # Hide y-axis grid lines
                    showticklabels=False, # Hide y-axis tick labels
                    zeroline=False, # Hide the y=0 plane
                    title="" # Remove y-axis title
                ),
                zaxis=dict(
                    showgrid=False,  # Hide z-axis grid lines
                    showticklabels=False, # Hide z-axis tick labels
                    zeroline=False, # Hide the z=0 plane
                    title="" # Remove z-axis title
                )
            )
        )
    # Otherwise the figure is 2D, so modify it another way
    else:
        fig.update_xaxes(
            showgrid        =   False,
            ticks           =   "",
            showticklabels  =   False,
            zeroline        =   False,
            title           =   "",  
        )
        fig.update_yaxes(
            showgrid        =   False,
            ticks           =   "",
            showticklabels  =   False,
            zeroline        =   False,
            title           =   "",
        )        


def set_background_color( 
        fig, 
        background_color="white",
        font_color=None,
        blank_background=True,
    ):
    """
    Set the background color of a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objs.Figure
        A Plotly Figure object containing a 3D scatter or surface plot.
    background_color : str (default="white")
        The desired background color. 
    font_color : str (default=None)
        - The desired font color for the text (legend, axes, titles, etc.).
        - If None, then the function will check if the provided background color is dark or light
          and set the font color accordingly (white for dark backgrounds, black for light backgrounds).
          The test for darkness is `oat_python.plot.is_dark_color(background_color)`.
    blank_background : bool (default=True)
        If True, the function will call :func:`blank_background` to remove grid lines, tick labels, and axis titles from the figure.

    Returns
    -------
    None
        The function modifies the input figure in place.

    Notes
    -----
    - Colors can be specified using any format accepted by Plotly (e.g., "crimson", "#FF0000", "rgb(255,0,0)").
    - **For dark plots** the user may wish to combine this method with ``fig.update_layout(template="plotly_dark")`` to ensure that axis lines and text are also rendered in light colors.

    Example
    -------
    See the :ref:`styling_3d_gallery` gallery for examples and styling.
    """

    # Set the background color of the 3D scene
    # ---------------------------------------------------------------
    fig.update_layout(
        paper_bgcolor=background_color,  # The area around the plot
        plot_bgcolor=background_color,    # The area inside the axes
    )

    # Set the background color of the plot itself
    # ---------------------------------------------------------------
    fig.update_scenes(
        xaxis_backgroundcolor=background_color,
        yaxis_backgroundcolor=background_color,
        zaxis_backgroundcolor=background_color,
    )


    # Set the font color
    # ---------------------------------------------------------------

    if font_color is None:
        if is_dark_color(background_color):            
            font_color = "white"
        else:
            font_color = "black"
    fig.update_layout(
        font=dict(
            color=font_color  # Sets the default font color for the entire figure
        )        
    )

    # Remove gridlines
    # ---------------------------------------------------------------
    if blank_background:
        oat_python.plot.blank_background(fig)
        





def is_dark_color(color, threshold=128):
    """
    Determines if a color is "dark" based on its perceived brightness.
    Supports hex, color names, and 'rgb(r,g,b)' strings.

    Calculates brightness using the formula:
    brightness = 0.2126*R + 0.7152*G + 0.0722*B
    where R, G, and B are the red, green, and blue components (0-255).

    Parameters
    ----------
    color : str
        The color to evaluate. Can be a hex string (e.g., '#RRGGBB'), a color name (e.g., 'red'), or an 'rgb(r,g,b)' string.
    threshold : int, optional
        Brightness threshold (0-255) below which a color is considered dark. Default is 128.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_dark_color('#000000')
    True
    >>> is_dark_color('#FFFFFF')
    False
    >>> is_dark_color('#0000FF')
    True
    >>> is_dark_color('#FFFF00')
    False
    >>> is_dark_color('white')
    False
    >>> is_dark_color('rgb(50,50,50)')
    True
    >>> is_dark_color('rgb(200,200,200)')
    False   
    """
    try:
        if isinstance(color, str) and color.startswith('rgb'):
            # Parse 'rgb(r,g,b)' string
            nums = re.findall(r'\d+', color)
            r, g, b = [int(x) for x in nums]
        else:
            # Use matplotlib for hex and color names
            r, g, b = mcolors.to_rgb(color)
            r, g, b = [int(255*x) for x in (r, g, b)]
    except Exception:
        print(f"Warning: Could not parse color {color}. Defaulting to light.")
        return False

    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return brightness < threshold        