//! [Open applied topology (OAT)](https://openappliedtopology.github.io) is a library for fast, user-friendly algebra and topology. OAT has 
//! 
//! - a user-friendly frontend for Python users, called [oat_python](https://github.com/OpenAppliedTopology/oat_python)
//! - a fast backend written in Rust, called [oat_rust](https://github.com/OpenAppliedTopology/oat_rust) 
//! 
//! This package contains the source code for [oat_python](https://crates.io/crates/oat_python).
//! 
//! # Contents
//! 
//! 
//! - [Welcome!](#welcome)
//! - [Community](#community)
//! - [Values](#values)
//! - [Mission](#mission)
//! - [Get started](#get-started)
//! 
//! # Welcome!
//! 
//! Welcome!  This package is [oat_python](https://crates.io/crates/oat_python), part of the Open Applied Topology ecosystem.  It provides powerful tools for applied topology, including
//! 
//! - Persistent homology
//! - Simplicial complexes
//! - Homological algebra
//! - Cycle optimization
//! - Interactive 2d and 3d visualization
//! 
//! 
//! # Community
//! 
//! OAT is by and for the open source community.  Reach out to the developers at [openappliedtopology@gmail.com](mailto:openappliedtopology@gmail.com) if you
//! - Need help getting started
//! - Wish for a missing feature
//! - Want to try coding
//! 
//! A collaboration of 20 research centers at colleges, universities, private, and public organizations support OAT's
//! development. The founding developers are Princton University, Macalester College, and the University of Delaware
//! The National Science Foundation granted seed funding for OAT in
//! 2019 under the [ExHACT]((https://www.nsf.gov/awardsearch/showAward?AWD_ID=1854748&HistoricalAwards=false))
//! project, and [Pacific Northwest National Laboratory](https://www.pnnl.gov/) (PNNL) provides continuing financial
//! support.  PNNL now coordinates development.  See <span style="color: SteelBlue;">[here](./ATTRIBUTIONS.md)</span>
//! for further details.
//! 
//! # Values
//! 
//! Our [shared values](https://github.com/OpenAppliedTopology/oat_python/blob/main/CODE_OF_CONDUCT.md) are
//! 
//! - Inclusion
//! - Respect, and 
//! - A shared passion to expand human knowledge, through algebraic topology
//! 
//! 
//! # Mission
//! 
//! **Performance**
//!     
//! OAT is a first-class solver for cutting-edge applications.  It is ideally suited to large, sparse data sets.
//!     The core library is written in Rust, a low-level systems programming language designed for safety and performance.
//!     High-level wrappers are available in Python. 
//! 
//! 
//! **Reliability**
//! 
//! OAT has more unit tests than type definitions and function definitions, combined.
//!   Its modular design enables end users to write their own checks for correctness, with ease.
//!   The library inherits strong safety guarantees from the the Rust compiler.
//! 
//! 
//! **Transparency**
//! 
//! OAT documentation emphasizes clarity and accessibility for users with all backgrounds.  It includes more than 180 working examples, and describes both code and underlying mathematical concepts in detail. 
//! [Online Jupyter notebook tutorials](crate::tutorials) illustrate how to combine multiple tools into larger applications.
//! The platform's modular design breaks large solvers into basic components, which it exposes to the user for inspection.  In addition, the library provides powerful methods to inspect and analyze objects, consistent with the way humans naturally think about problems; for example, you can look up rows and columns of boundary matrices using *cubes*, *simplices*, or *cells* as keys.
//!   
//! 
//! **Modularity**
//! 
//!   OAT reduces complex problems to the same basic building blocks that topologists use when writing on a chalk board. Users can mix and match those blocks with confidence, using a simple, streamlined interface.  They can even create new components that work seemlessly with the rest of the library, including coefficient rings, sparse matrix data structures, and customized filtrations on simplicial complexes.
//! 
//! 
//! # Contributing
//! 
//! We welcome contributions from everyone.
//! That's the "Open" part of "Open Applied Topology"!
//! Please see [CONTRIBUTING.md](https://github.com/OpenAppliedTopology/oat_python/blob/main/CONTRIBUTING.md)
//! for help getting started.
//! 
//! # Get Started
//! 
//! The oat_python package is written in Rust and Python. This is the documentation page for the Rust API, which you can explore by making selections either from the menu on the left or from the list of modules below.
//!
//! [**Python users and new developers:** see README.md on the package GitHub repository for tips to get started!](https://github.com/OpenAppliedTopology/oat_python/blob/main/README.md)



pub mod vietoris_rips;
pub mod dowker;
pub mod export;
// pub mod simplex_filtered;
pub mod import;
pub mod utilities;
pub mod zigzag;

// ------------


use oat_rust::topology::simplicial::simplices::weighted::WeightedSimplex;
use vietoris_rips::VietorisRipsBoundaryMatrixOverQ;
use dowker::DowkerComplexDifferentialUmatch;
use itertools::Itertools;
use ordered_float::OrderedFloat;
// use dowker::UmatchPyDowkerRational;
use pyo3::prelude::*;

// use simplex_filtered::WeightedSimplexPython;
// use simplex_filtered::BarcodePyWeightedSimplexRational;
// use simplex_filtered::BarPyWeightedSimplexRational;
// use simplex_filtered::MinimalCyclePyWeightedSimplexRational;
use import::import_sparse_matrix;

use vietoris_rips::{DifferentialUmatchVietorisRipsPython};
use vietoris_rips::VietorisRipsComplexPython;
use export::*;
use sprs::CsMat;
use sprs::CsMatBase;

// use crate::clique_filtered::__pyo3_get_function_persistent_homology_vr;
// use crate::clique_filtered::__pyo3_get_function_persistent_homology_vr_optimized;
// use crate::dowker::__pyo3_get_function_homology_basis_from_dowker;
// use crate::dowker::__pyo3_get_function_transpose_listlist;
// use crate::dowker::__pyo3_get_function_unique_row_indices;
// use crate::dowker::__pyo3_get_function_unique_rows;
// use crate::clique_filtered::persistent_homology_vr;
// use crate::clique_filtered::persistent_homology_vr_optimized;
// use crate::dowker::homology_basis_from_dowker;
// use crate::dowker::transpose_listlist;
// use crate::dowker::unique_row_indices;
use crate::dowker::{unique_rows,unique_row_indices, transpose_listlist};
use crate::vietoris_rips::{GeneralizedMatchingMatrix, LaplacianMatrix, ChangeOfBasisMatrix, ChangeOfBasisMatrixInverse, SubmatrixIndexTool, VectorIndexTool, WeightedSimplexPython};
use crate::zigzag::zigzag_homology;


// use pyo3::types::PyDict;

use num::{rational::Ratio, ToPrimitive};




/// This module contains the objects and functions in ``oat_python`` that have
/// been exported from Rust. It is called ``core`` because Rust provides many of
/// the most basic building blocks of this library.
/// **Rust does not need to be installed to use this module.**
/// 
/// This module contains the following submodules:
/// 
/// - ``vietoris_rips``: Contains the Vietoris-Rips complex and related objects. For details see the :ref:`Vietoris-Rips <vietoris-rips-section>`.
/// - ``dowker``: Contains the Dowker complex and related objects. For details see the :ref:`Hypergraph <hypergraph-section>`.
///
/// 
#[pymodule(name="core")]
fn oat_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<WeightedSimplexPython>()?; 
    // m.add_class::<BarPyWeightedSimplexRational>()?;    
    // m.add_class::<BarcodePyWeightedSimplexRational>()?;
    // m.add_class::<MinimalCyclePyWeightedSimplexRational>()?; 

    register_child_module_vietoris_rips(m)?; 
    register_child_module_dowker(m)?;    
    Ok(())
}







fn register_child_module_vietoris_rips(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new(parent_module.py(), "vietoris_rips")?;

    child_module.add_class::<DifferentialUmatchVietorisRipsPython>()?;
    child_module.add_class::<VietorisRipsComplexPython>()?;    
    child_module.add_class::<VietorisRipsBoundaryMatrixOverQ>()?;    
    child_module.add_class::<SubmatrixIndexTool>()?;   
    child_module.add_class::<VectorIndexTool>()?;       
    child_module.add_class::<WeightedSimplexPython>()?;        
    child_module.add_class::<GeneralizedMatchingMatrix>()?; 
    child_module.add_class::<ChangeOfBasisMatrix>()?;          
    child_module.add_class::<ChangeOfBasisMatrixInverse>()?;     
    child_module.add_class::<LaplacianMatrix>()?;                               


    parent_module.add_submodule(&child_module)
}

fn register_child_module_dowker(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new(parent_module.py(), "dowker")?;

    child_module.add_class::<DowkerComplexDifferentialUmatch>()?;
    child_module.add_function(wrap_pyfunction!(unique_rows, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(unique_row_indices, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(transpose_listlist, &child_module)?)?;  

    child_module.add_function(wrap_pyfunction!(zigzag_homology, &child_module)?)?;  

    parent_module.add_submodule(&child_module)
}

