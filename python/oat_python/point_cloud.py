"""
This module contains functions to generate a variety of point clouds,
including random points on disks, circles, spheres, and tori.

**Persistent homology and cohomology** 

For the persistent (co)homology of point clouds, see the :ref:`Vietoris Rips section <vietoris-rips-section>`.
"""


import copy
import os
import importlib.resources
from typing import Tuple
import numpy as np
import sklearn
from sklearn.neighbors import radius_neighbors_graph

import scipy




def disk(n_points, radius, random_seed=None):
    """
    Randomly samples points from the disk of given radius centered at the origin in the Euclidean plane.

    Parameters
    ----------
    n_points : int
        The number of points to sample from the disk.
    radius : float
        The radius of the disk.
    random_seed : int or None (optional, default=None)
        If provided, sets the random seed for reproducibility.

    Returns
    -------
    np.ndarray
        An `N x 2` array of points sampled from the disk.
    """
    # Set the random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random angles
    angles = np.random.uniform(0, 2*np.pi, n_points)
    
    # Generate random radii within the disk
    radii = np.sqrt(np.random.uniform(0, radius**2, n_points))
    
    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    return np.column_stack((x, y))


def two_circles():
    """
    Points evenly spaced along two tangent circles in the Euclidean plane.
    """
    theta  = np.linspace(0, 2*np.pi, 100)
    points = np.zeros((100,2))
    points[:,0] = np.cos(theta)
    points[:,1] = np.sin(theta)
    cloud2 = copy.deepcopy(points)
    cloud2[:,0] += 1
    points = np.concatenate((points,cloud2),axis=0)
    return points

def torus_curve( n_points=100 , inner_radius=0.3, outer_radius=1, angle_initial=0, nturns=1, ):
    """
    Return a sequence of points that curves around a torus `nturns` times.
    """
    r           = inner_radius
    R           = outer_radius
    T           =   np.linspace(0, 2 * np.pi,          n_points) # major angle
    t           =   np.linspace(0, 2 * np.pi * nturns, n_points) + angle_initial * np.pi # minor angle
    x           =   ( R + r * np.cos( t ) ) * np.cos( T )
    y           =   ( R + r * np.cos( t ) ) * np.sin( T )
    z           =   r * np.sin( t )
    points      =   np.zeros((n_points,3))
    points[:,0]=x; points[:,1]=y; points[:,2]=z;
    return points


def torus(
        radius_outer = 4, 
        radius_inner = 1, 
        n_points_outer = 60, 
        n_points_inner = 20, 
        repeat_last = True,
    ):
    """
    Returns an `N x 3` array of points sampled from a torus in 3D Euclidean space.

    :param radius_outer: the larger radius (default = 4)
    :param radius_inner: the smaller radius (default = 1)
    :param n_points_outer: number of points to sample from the larger circle (default = 60)
    :param n_points_inner: number of points to sample from the smaller circle (default = 20)
    :param repeat_last: determines whether the first and last rows (respectively, columns) of each `x, y, z` matrix are equal; defualt = `True`
    """
    phi_values = np.linspace(0, 2 * np.pi, n_points_outer, endpoint=False)
    theta_values = np.linspace(0, 2 * np.pi, n_points_inner, endpoint=False)
    if repeat_last:
        phi_values = np.append(phi_values, 0)
        theta_values = np.append(theta_values, 0)
    phi, theta = np.meshgrid(phi_values, theta_values)
    phi = phi.flatten()
    theta = theta.flatten()

    x = (radius_outer + radius_inner * np.cos(theta)) * np.cos(phi)
    y = (radius_outer + radius_inner * np.cos(theta)) * np.sin(phi)
    z = radius_inner * np.sin(theta)

    return np.column_stack((x, y, z))

def sphere_or_slice_spiral( n_points=50, embedding_dim=3, noise_scale=1, xmax=1, random_seed=None ):
    """
    A sample of points for a sphere intersected with a halfspace of form x ≤ xmax, formatted as an N x 3 array.

    Points are sampled using a "spiral" method; see the source code for details.

    Parameters
    ----------
    n_points : int
        Number of points in the points.
    embedding_dim : int
        Embedding dimension of the points, defaults to 3.
    noise_scale : float
        Points are generated in a deterministic fashion, then modified by adding a small amount of noise drawn iid from the unit cube of size [0,noise_scale]^n.
    xmax : float
        Determines how far on the x axis you go; xmax=0 returns a hemisphere.
    random_seed : int or None (default 0)
        A random seed for the generation of noise, for reproducibility. If None, the random number generator is not seeded.
    """

    # handle the edge case of 0 points
    if n_points == 0:
        return np.zeros((0,3))    

    if not random_seed is None:
        np.random.seed( random_seed )

    theta   = np.linspace(0, 9 * 2 * np.pi, n_points)
    x       = np.linspace(-1, xmax, n_points)
    points = np.zeros((n_points, embedding_dim))
    points[:,0] += x
    points[:,1] += np.cos(theta) * (1-x**2)
    points[:,2] += np.sin(theta) * (1-x**2)
    points += np.random.rand(n_points, embedding_dim) * noise_scale
    
    return points
    


def half_dome( n_points=50, noise_scale=1 ):
    """
    Samples points evenly from the half of the Euclidean 2-sphere that lies between x = -1 and x = 0.

    This function is just a wrapper for ``sphere_or_slice(n_points=n_points, xmin=-1, xmax=0) + noise``.

    Parameters
    ----------
    n_points : int, optional
        Number of points to sample. Default is 50.
    noise_scale : float, optional
        Standard deviation of the random noise added to each coordinate. Default is 1.

    Returns
    -------
    points : numpy.ndarray, shape (n_points, 3)
        Array of sampled points on the half-sphere, each row is a 3D point [x, y, z].
    """
    points = sphere_or_slice( n_points=n_points, randomize=True, xmin=-1.0, xmax=0.0)
    points += np.random.rand( n_points, 3 ) * noise_scale
    return points   


# A function to generate random points on a sphere, sliced between the plane x=xmin and x=xmax
def sphere_or_slice(  n_points=1, xmin=-1.0, xmax=1.0, randomize=False, random_seed=None, ):
    """
    Samples (approximately) evenly distributed points from the portion of the unit sphere in Euclidean 3-space
    that lies between the planes ``x = xmin`` and ``x = xmax``, using the Fibonacci spiral method.

    Parameters
    ----------
    n_points : int, optional
        Number of points to sample. Default is 1.
    xmin : float, optional
        Lower threshold for x coordinates (inclusive). Default is -1.0.
    xmax : float, optional
        Upper threshold for x coordinates (inclusive). Default is 1.0.
    randomize : bool, (optional, default=False)
        If True, adds randomization to the initial condition for the spiral. Default is True.
    random_seed : int or None (default None)
        A random seed for the generation of noise, for reproducibility. If None, the random number generator is not seeded.

    Returns
    -------
    points : numpy.ndarray, shape (n_points, 3)
        Array of sampled points on the sphere slice, each row is a 3D point [x, y, z].

    Notes
    -----
    - The method uses a Fibonacci Lattice to distribute points approximately evenly on the sphere.
    - The slice is defined by the region where ``xmin <= x <= xmax``.
    - If ``randomize`` is True, the spiral is randomly rotated for more uniformity.
    - If ``random_seed`` is provided, the randomization is reproducible.
    """
    # handle the edge case of 0 points
    if n_points == 0:
        return np.zeros((0,3))

    rnd = 1.
    if randomize:
        if not random_seed is None:
            np.random.seed( random_seed )
        rnd = np.random.random() * n_points

    points = []
    increment_y   = (xmax - xmin)/n_points
    increment_phi = np.pi * (3. - np.sqrt(5.))

    for i in range( n_points ):
        x = xmin + i * increment_y # ((i * offset) - 1) + (offset / 2) #  #
        r = np.sqrt(1 - x**2)
        phi = ((i + rnd) % n_points) * increment_phi
        y = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x, y, z])

    return np.array(points)        



def annulus( n_points=1, inner_radius=1, outer_radius=2, random_seed=None ):
    """
    Samples points uniformly from an annulus (ring-shaped region) in the plane.

    The annulus is defined as the region between two concentric circles centered at the origin,
    with inner radius `inner_radius` and outer radius `outer_radius`. Points are sampled uniformly with respect
    to area.

    Parameters
    ----------
    n_points : int, optional
        Number of points to sample. Default is 1.
    inner_radius : float, optional
        Inner radius of the annulus. Default is 1.
    outer_radius : float, optional
        Outer radius of the annulus. Default is 2.
    random_seed : int or None, optional
        If provided, sets the random seed for reproducibility.

    Returns
    -------
    points : numpy.ndarray, shape (n_points, 2)
        Array of sampled points in the annulus, each row is a 2D point [x, y].

    Notes
    -----
    - The sampling is uniform with respect to area, not radius.
    - If `inner_radius` equals `outer_radius`, all points will lie on the circle of that radius.
    - If `random_seed` is provided, the randomization is reproducible.
    """
    points = np.zeros((n_points,2))

    if not random_seed is None:
        np.random.seed( random_seed )

    scale_parameter     =   inner_radius**2-outer_radius**2
    for i in range(n_points):
        theta       =   2 * np.pi * np.random.rand()
        rnd         =   np.random.rand()
        rho         =   np.sqrt( rnd * scale_parameter + outer_radius**2 );
        points[i][0]    =   rho * np.cos(theta)
        points[i][1]    =   rho * np.sin(theta)

    return points



def circle(n_points=10, radius=1, mode="uniform", random_seed=None):
    """
    Generate points on a circle of given radius.

    Parameters
    ----------
    n_points : int (optional, default=10)
        Number of points to sample.
    radius : float (optional, default=1)
        Radius of the circle.
    mode : {'random', 'uniform'} (optional, default='uniform')

        - 'random': Points are sampled by drawing angles uniformly at random from [0, 2π).
        - 'uniform': Points are placed at equally spaced angles from 0 to 2π (not including endpoint).

        Default is 'uniform'.
    random_seed : int or None (optional, default=None)
        If provided and mode is 'random', sets the random seed for reproducibility.

    Returns
    -------
    points : numpy.ndarray, shape (n_points, 2)
        Array of sampled points on the circle, each row is a 2D point [x, y].

    Notes
    -----
    - In 'random' mode, the sampling is uniform with respect to angle, not arc length.
    - In 'uniform' mode, the points are spaced uniformly in angle, so the first and last points may coincide if `n_points` divides 2π evenly.
    - If `random_seed` is provided, the randomization is reproducible (only for 'random' mode).

    """
    points = np.zeros((n_points, 2))

    if mode == "random":
        if random_seed is not None:
            np.random.seed(random_seed)
        theta = 2 * np.pi * np.random.rand(n_points)
    elif mode == "uniform":
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    else:
        raise ValueError("mode must be 'random' or 'uniform'")

    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)

    return points


def stanford_dragon():
    """
    Load a point cloud of 1000 points, sampled from the Stanford Dragon.

    The points are returned as a numpy array of shape (1000, 3).

    Notes
    -----
    The data file is saved to `oat_python/data/stanford_dragon.txt`.
    """     
    
    with importlib.resources.files('oat_python.data').joinpath('stanford_dragon.txt').open('r') as f:
        points = np.loadtxt(f, delimiter=' ')
    points                      =   points[:,[0,2,1]]
    return points  

    # DEPRECATED / ok to delete:
    # # Get the path relative to this file's location, going up two levels to reach project root
    # root_dir        =   os.path.dirname(
    #                         os.path.dirname(
    #                             os.path.dirname(
    #                                 os.path.abspath(__file__)
    #                             )
    #                         )
    #                     )
    # data_path       =   os.path.join(root_dir, 'data', 'stanford_dragon.txt')
    # with open(data_path, 'r') as f:
    #     points      = np.loadtxt(f, delimiter=' ')
    # points          = points[:, [0, 2, 1]]
    # return points    
      