//!  Filtered clique (Vietoris-Rips) complexes

use derive_getters::Dissolve;
use num::Signed;
use oat_rust::algebra::chain_complexes::ChainComplex;
use oat_rust::algebra::matrices::operations::umatch::row_major::{self, Umatch};
use oat_rust::algebra::matrices::types::transpose::OrderAntiTranspose;
use oat_rust::algebra::matrices::types::two_type::TwoTypeMatrix;
use oat_rust::topology::simplicial::boundary;
use oat_rust::utilities::sequences_and_ordinals::BijectiveSequence;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList};
use pyo3::wrap_pyfunction;
// use pyo3_log;
use pyo3::types::{PyDict, PyTuple};
use oat_rust::algebra::matrices::operations::umatch::differential::{self, DifferentialUmatch};
use oat_rust::algebra::matrices::operations::MatrixOracleOperations;
use oat_rust::algebra::matrices::display::print_indexed_rows;
use oat_rust::algebra::matrices::types::third_party::IntoCSR;

use oat_rust::algebra::chain_complexes::barcode::{Bar, Barcode};
use oat_rust::algebra::vectors::entries::{KeyValGet, KeyValNew, KeyValSet};
use oat_rust::algebra::matrices::query::{MatrixAlgebra, MatrixOracle};
use oat_rust::algebra::rings::traits::{SemiringOperations, RingOperations, DivisionRingOperations};
use oat_rust::algebra::rings::types::native::{FieldFloat64, FieldRationalSize, RingOperatorForNativeRustNumberType};
use oat_rust::utilities::iterators::general::{RequireStrictAscent, RequireStrictAscentWithPanic};
use oat_rust::utilities::order::{is_sorted_strictly, JudgePartialOrder, ReverseOrder};
use oat_rust::utilities::order::{OrderOperatorAuto, OrderOperatorByKey, OrderOperatorByKeyCustom, IntoReverseOrder};
use oat_rust::algebra::vectors::operations::VectorOperations;

use oat_rust::topology::simplicial::simplices::unweighted::Simplex;
use oat_rust::topology::simplicial::simplices::weighted::{self, OrderOperatorTwistWeightedSimplex};
use oat_rust::utilities::optimization::minimize_l1::minimize_l1;
use oat_rust::utilities::order::JudgeOrder;
use oat_rust::utilities::iterators::general::PeekUnqualified;
use oat_rust::topology::simplicial::simplices::weighted::WeightedSimplex;
use oat_rust::topology::simplicial::from::graph_weighted::VietorisRipsComplex;
use serde::de;

use crate::dowker::unique_row_indices;
use crate::export::{Export, ForExport, IntoDataframeFormat, IntoScipyCsrFormat, IntoVecOfPyTuples};
use crate::import::import_sparse_matrix;
// use crate::simplex_filtered::{WeightedSimplexPython, BarPyWeightedSimplexRational, BarcodePyWeightedSimplexRational, };

use itertools::{Diff, Itertools};
use num::rational::Ratio;
use num::Zero;
use ordered_float::OrderedFloat;
use sprs::{CsMatBase, TriMatBase};


use rand_distr::{Distribution, Normal, NormalError};
use rand::thread_rng;

use std::collections::HashMap;
use std::f32::consts::E;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Cloned;
use std::ops::Sub;
use std::sync::Arc;
use std::time::Instant;

type FiltrationValue     =   OrderedFloat<f64>;
type RingElement         =   Ratio<isize>;








#[pyclass(name = "WeightedSimplex")]
pub struct WeightedSimplexPython {
    simplex: Vec<u16>,
    weight: f64,
}

#[pymethods]
impl WeightedSimplexPython {


    /// Create a new `WeightedSimplex` instance
    /// 
    /// # Arguments
    /// 
    /// - `vertices`: a vector of vertices, e.g. `[0, 1, 2]`. Vertices must be strictly sorted.
    /// - `weight`: a `f64` representing the weight of the simplex, e.g. `1.0`.
    /// 
    /// # Returns
    /// 
    /// A new `WeightedSimplex` instance if the vertices are strictly sorted, otherwise an error.
    #[new]
    pub fn new(simplex: Vec<u16>, weight: f64) -> PyResult< Self > {
        if is_sorted_strictly(&simplex, &OrderOperatorAuto) {
            return Ok(Self { simplex, weight })
        } else {
            return Err(PyTypeError::new_err(format!(
                "Vertices {:?} are not strictly sorted.",
                simplex
            )));
        }
        
    }


    /// Returns the vertices of the simplex
    pub fn simplex<'py>(&self, py: Python<'py>) ->PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.simplex.clone())
    }

    /// Returns the weight of the simplex
    pub fn weight(&self) -> f64 {
        self.weight.clone()
    }

    /// Returns the dimension of the simplex
    pub fn dimension(&self) -> isize {
        self.simplex.len() as isize - 1
    }


    fn __repr__(&self) -> String {
        format!(
            "<WeightedSimplex simplex={} | weight={:.6}>",
            format!("{:?}", self.simplex),
            self.weight
        )
    }

    fn __str__(&self) -> String {
        let verts = self.simplex.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "Simplex: ({})   Weight: {:.6}",
            verts,
            self.weight
        )
    }
}

// Conversion from Rust WeightedSimplex<OrderedFloat<f64>> to Python wrapper
impl From<&WeightedSimplex<OrderedFloat<f64>>> for WeightedSimplexPython {
    fn from(ws: &WeightedSimplex<OrderedFloat<f64>>) -> Self {
        Self {
            simplex: ws.vertices.clone(),
            weight: ws.weight.into_inner(),
        }
    }
}


// Conversion from Rust WeightedSimplex<OrderedFloat<f64>> to Python wrapper
impl From<WeightedSimplex<OrderedFloat<f64>>> for WeightedSimplexPython {
    fn from(ws: WeightedSimplex<OrderedFloat<f64>>) -> Self {
        Self {
            simplex: ws.vertices,
            weight: ws.weight.into_inner(),
        }
    }
}







// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================








#[pyclass(name = "VietorisRipsComplex")]
pub struct VietorisRipsComplexPython{
    inner_vietoris_rips_complex:    Arc<
                                        VietorisRipsComplex<
                                            Arc< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
                                            RingOperatorForNativeRustNumberType<RingElement>
                                        >
                                    >    
}

#[pymethods]
impl VietorisRipsComplexPython{ 
    /// Construct a Vietoris-Rips complex over the field of rational numbers
    /// 
    /// # Arguments
    /// 
    /// `dissimilarity_matrix`: a sparse dissimilarity matrix stored as a Scipy sparse CSR
    ///   - missing entries will be treated as edges that never enter the filtration
    ///   - diagonal entries will be treated as vertex birth times. If `dissimilarity_matrix[i,i]` is structurally zero then vertex `i` is not included in the filtration.
    ///   - the matrix must be symmetric
    ///   - if row `i` contains any structural nonzero entry, then entry `(i,i)` must be structurally nonzero, and the smallest of all structural nonzero entries in that row
    /// 
    /// 
    /// # Returns
    /// 
    /// A `VietorisRipsBoundaryMatrixOverQ`
    /// 
    /// # Errors
    /// 
    /// Returns an error if 
    /// - `dissimilarity_matrix` is not symmetric
    /// - there exists an `i` such that entry `[i,i]` is not explicitly stored, but some other entry in row `i` *is* explicitly stored.
    ///   this is because we regard missing entries as having infinite value, rather than zero.
    /// - there exists an `i` such that entry `[i,i]` is strictly greater than some other entry in row `i`
    #[new]
    pub fn new<'py>(
            dissimilarity_matrix:       &Bound<'py, PyAny>,        
        ) 
    ->  PyResult<VietorisRipsComplexPython>
    {

        let dissimilarity_matrix = import_sparse_matrix(dissimilarity_matrix)?;
        let n_points = dissimilarity_matrix.rows();
        let dissimilarity_matrix = Arc::new( dissimilarity_matrix );                   

        // define the ring operator
        let ring_operator = FieldRationalSize::new();
        // define the chain complex


        let chain_complex_data = match 
            VietorisRipsComplex::new( 
                dissimilarity_matrix.clone(), 
                n_points, 
                ring_operator 
        )        
        {
            Ok(chain_complex_data) => {
                chain_complex_data
            },
            Err(e) => {
                return Err(PyTypeError::new_err(format!("\nError constructing Vietoris-Rips complex: the dissimilarity matrix is not symmetric. \
                                                         Entry {:?} equals {:?} but entry {:?} equlas {:?}. This message is generated by by OAT.", 
                                                        e,
                                                        dissimilarity_matrix.get(e.0, e.1),
                                                        (e.1, e.0),
                                                        dissimilarity_matrix.get(e.1, e.0),
                                                    )));
            },
        };     
        let chain_complex = Arc::new( chain_complex_data );
        return Ok( VietorisRipsComplexPython{ inner_vietoris_rips_complex: chain_complex } ) 
    }     


    /// Returns the simplices of the Vietoris-Rips complex in sorted order
    /// 
    /// Simplices are sorted first by dimension, then by filtration value, then lexicographic order.
    pub fn simplices_for_dimensions(
            &self, 
            py:             Python<'_>,            
            mut dimensions: Vec<isize>
        ) -> PyResult<PyObject> 
    {      
        dimensions.sort();
        let simplices  =   dimensions.into_iter()
                                .dedup()
                                .map(
                                    |dimension|
                                    self.inner_vietoris_rips_complex.cliques_in_boundary_matrix_order_fixed_dimension(dimension)
                                )
                                .flatten()
                                .collect_vec();
        return simplices.into_dataframe_format(py)
    }   


    /// Returns the filtration value of a simplex
    /// 
    /// # Arguments
    /// 
    /// - `simplex`: a strictly sorted list of vertices, e.g. `[0,1,2]`
    /// 
    /// # Returns
    /// 
    /// A `f64` representing the filtration value of the simplex, or an error if the Vietoris-Rips complex does not contain this simplex.
    pub fn filtration_value(
            &self, 
            simplex: Vec<u16>
        ) -> PyResult<f64> 
    {
        // ensure simplex is strictly sorted
        if ! is_sorted_strictly(&simplex, &OrderOperatorAuto) {
            return Err(PyTypeError::new_err(format!("Simplex {:?} is not strictly sorted.", &simplex)));
        }
        let filtration_value = self.inner_vietoris_rips_complex.filtration_value_for_clique(&simplex);
        match filtration_value {
            Ok(d) => Ok(d.into_inner()),
            Err(_) => Err(PyTypeError::new_err(format!("The input {:?} is not a valid simplex in this Vietoris-Rips complex.", &simplex))),
        }
    }



    /// Returns a :term:`matrix oracle` for the boundary matrix of the Vietoris-Rips complex
    pub fn boundary_matrix_oracle(&self) -> VietorisRipsBoundaryMatrixOverQ {
        VietorisRipsBoundaryMatrixOverQ{ 
            inner_vietoris_rips_complex: self.inner_vietoris_rips_complex.clone() 
        }
    }




    /// Returns a :term:`matrix oracle` for the Laplacian matrix of the Vietoris-Rips complex
    pub fn laplacian_matrix_oracle(
            &self, 
            // deformation_coefficient: Option<f64>,
        ) -> PyResult< LaplacianMatrix >
    {
        // let deformation_coefficient = deformation_coefficient.unwrap_or(1.0);
        let deformation_coefficient = 0.0;

        
        let ring_operator = FieldFloat64::new();
        let boundary_matrix_data = match 
            VietorisRipsComplex::new( 
                self.inner_vietoris_rips_complex.dissimilarity_matrix.clone(),
                self.inner_vietoris_rips_complex.dissimilarity_matrix_size, 
                ring_operator 
        )        
        {
            Ok(chain_complex_data) => {
                chain_complex_data
            },
            Err(e) => {
                return Err(PyTypeError::new_err(format!("\nError constructing Vietoris-Rips complex: the dissimilarity matrix is not symmetric. \
                                                         Entry {:?} equals {:?} but entry {:?} equlas {:?}. This message is generated by by OAT.", 
                                                        e,
                                                        self.inner_vietoris_rips_complex.dissimilarity_matrix.get(e.0, e.1),
                                                        (e.1, e.0),
                                                        self.inner_vietoris_rips_complex.dissimilarity_matrix.get(e.1, e.0),
                                                    )));
            },
        };  
        let boundary_matrix = Arc::new( boundary_matrix_data );

        Ok( LaplacianMatrix{
            boundary_matrix,
            deformation_coefficient,
        } )
    }






    /// Returns a `VectorIndexTool`, which provides an ergonomic way to translate between simplex-based and number-based indexing
    /// 
    /// # Arguments
    /// 
    /// - `dimensions`: a list of simplex dimensions (defaults to an empty list)
    /// 
    /// # Returns
    /// 
    /// A `VectorIndexTool` object containing simplices of the specified dimensions.
    /// These simplex indices are sorted in the standard order: by dimension, then filtration value, then lexicographically.
    /// This is the same order used for the boundary matrix itself.
    #[pyo3(signature = (dimensions=None))]
    pub fn vector_index_tool(
            &self, 
            dimensions:    Option< Vec<isize> >,
        ) -> PyResult<VectorIndexTool> 
    {        

        let dimensions    =   dimensions.unwrap_or( vec![] );

        let simplex_getter = |dimensions: Vec<isize>| -> Vec<WeightedSimplex<OrderedFloat<f64>>> {
            let mut dimensions = dimensions.clone();
            dimensions.sort();
            dimensions.into_iter()
                .dedup()
                .map(
                    |dimension|
                    self.inner_vietoris_rips_complex.cliques_in_boundary_matrix_order_fixed_dimension(dimension)
                )
                .flatten()
                .collect_vec()
        };

        let simplices = simplex_getter(dimensions);

        let bimap   = BijectiveSequence::from_vec( simplices ).unwrap();

        Ok( VectorIndexTool{
            indices: bimap,
            inner_vietoris_rips_complex: self.inner_vietoris_rips_complex.clone(),
        } )
    }









    /// Returns a :class:`SubmatrixIndexTool` object containing ordered sequences of row and column indices
    /// 
    /// # Arguments
    /// 
    /// - `row_dimensions` (optional, defaults to the empty list)
    /// 
    ///   - A list of dimensions for the row indices
    ///   - If this list has form `[d1, d2, ...]`, then the row indices will
    ///     include every simplex of dimension `d1`, `d2`. 
    /// 
    /// - `column_dimensions` (optional, defaults to the empty list `[]`)
    /// 
    ///   - A list of dimensions for the column indices
    ///   - If this list has form `[d1, d2, ...]`, then the column indices will
    ///     include every simplex of dimension `d1`, `d2`. 
    /// 
    /// # Returns
    /// 
    /// A :class:`SubmatrixIndexTool` object containing the specified row and column indices.
    /// 
    /// - Indices are sorted first by dimension, then by filtration value, then lexicographically
    ///   (the same order used for the boundary matrix itself).
    /// - Duplicate dimensions are removed.
    #[pyo3(signature = (row_dimensions=None, column_dimensions=None))]
    pub fn submatrix_index_tool(
            &self, 
            row_dimensions:    Option< Vec<isize> >,
            column_dimensions: Option< Vec<isize> >,
        ) -> PyResult<SubmatrixIndexTool> 
    {        
        let chain_complex = self.inner_vietoris_rips_complex.clone();      

        let row_bimap = chain_complex.basis_vector_index_bimap_for_dimensions(
            row_dimensions.unwrap_or( vec![] )
        ).unwrap();
        let column_bimap = chain_complex.basis_vector_index_bimap_for_dimensions(
            column_dimensions.unwrap_or( vec![] )
        ).unwrap();

        Ok( SubmatrixIndexTool{
            row_indices: row_bimap,
            column_indices: column_bimap,
            inner_vietoris_rips_complex: self.inner_vietoris_rips_complex.clone(),
        } )
    }



}









// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================










/// Stores the row and column indices of a submatrix of the boundary matrix of a Vietoris-Rips complex
/// 
/// The primary use for this object is to provide an ergonomic / readable way to map between simplices
/// and row/column numbers of a submatrix.
/// 
/// **Caution**
/// 
/// This object is not an Oracle; it stores the row and column indices in memory. If the number of
/// simplices is large, this may lead to high memory usage.
#[pyclass]
pub struct VectorIndexTool{
    indices:                        BijectiveSequence<WeightedSimplex<OrderedFloat<f64>>>,
    inner_vietoris_rips_complex:    Arc<
                                        VietorisRipsComplex<
                                            Arc< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
                                            RingOperatorForNativeRustNumberType<RingElement>
                                        >
                                    >,
}


#[pymethods]
impl VectorIndexTool{ 


    fn __repr__(&self) -> String {
        format!(
            "<VectorIndexTool with {} unique simplices>",
            self.indices.len(),
        )
    }

    fn __str__(&self) -> String {
        format!(
            "VectorIndexTool with {} unique simplices",
            self.indices.len(),
        )
    }





    /// Sets the indices of the index tool
    /// 
    /// # Arguments
    /// 
    /// - `indices`: a list of list/tuples of nonnegative integers, e.g. `[[0,1,2], [1,2,3]]`.
    /// 
    ///   - Each inner list must be strictly sorted.
    ///   - The outer list can contain no duplicates.
    /// 
    /// # Errors
    /// 
    /// Returns an error if
    /// 
    /// - any of the inner lists is not strictly sorted, or
    /// - any of the simplices is not a valid simplex in the Vietoris-Rips complex
    /// - the outer list contains duplicate simplices
    pub fn set_simplices(
            &mut self, 
            indices: Vec<Vec<u16>>
        ) -> PyResult<()> 
    {
        let mut simplices = Vec::with_capacity(indices.len());

        for simplex in indices.iter() {
            let weighted = self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(simplex);
            match weighted {
                Ok(weighted_simplex) => {
                    simplices.push(weighted_simplex);
                },
                Err(_) => {
                    return Err(PyTypeError::new_err(format!("The input {:?} is not a valid simplex in this Vietoris-Rips complex.", &simplex)));
                },
            }            
        }

        match BijectiveSequence::from_vec( simplices ) {
            Ok(bijective_sequence) => {
                self.indices = bijective_sequence;
            },
            Err(e) => {
                return Err(PyTypeError::new_err( format!("User-provided indices include one or more duplicates, including {:?}.", e)));
            },
        }

        Ok(())
    }




    /// Returns the number of simplices in the index tool
    pub fn number_of_simplices(&self) -> usize {
        self.indices.len()
    }
 

    /// Returns the collection of indices as a dataframe
    /// 
    /// The kth row of the dataframe corresponds to the kth simplex in the index tool.
    pub fn simplices(&self, py: Python<'_>) -> PyResult<PyObject> {
        let row_indices: Vec<_> = self.indices.vec_elements_in_order().clone();
        row_indices.into_dataframe_format(py)
    }


    /// Returns True if the simplex tool contains `simplex` in its list of indices
    pub fn contains_simplex(
            &self, 
            simplex: Vec<u16>
        ) -> bool
    {
        match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                self.indices.has_ordinal_for_element(&weighted_simplex)
            },
            Err(_) => {
                false
            },
        }
    }

    
    /// Returns the index number of the given simplex
    /// 
    /// # Arguments
    /// 
    /// - `simplex`: a strictly sorted list of vertices, e.g. `[0,1,2]`
    /// 
    /// # Returns
    /// 
    /// An integer representing the location of the simplex within the list of simplices stored by the tool `simplex_0 .. simplex_n`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the Vietoris-Rips complex does not contain this simplex.
    pub fn index_number_for_simplex(
            &self, 
            simplex: Vec<u16>
        ) -> PyResult<usize> 
    {
        match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                let entry_number = self.indices.ordinal_for_element(&weighted_simplex);
                match entry_number {
                    Some(row_number) => Ok(row_number),
                    None => Err(PyTypeError::new_err(format!("Simplex {:?} is not a valid index for this VectorIndexTool.", &simplex))),
                }
            },
            Err(_) => {
                Err(PyTypeError::new_err(format!("The VectorIndexTool does not contain a simplex represented by {:?}.", &simplex)))
            },
        }
    }



    /// Returns the k-th simplex in the index tool
    ///
    /// # Arguments
    ///
    /// - `k`: an integer representing the row number in the submatrix
    ///
    /// # Returns
    ///
    /// A `WeightedSimplexPython` representing the simplex at the given row number
    /// 
    /// # Errors
    /// 
    /// Returns an error if `k` is greater than or equal to the number of indices in the tool. 
    pub fn simplex_for_index_number(
            &self, 
            k: usize
        ) -> PyResult<WeightedSimplexPython> 
    {
        match self.indices.element_result_for_ordinal(k) {
            Ok(simplex) => Ok( (&simplex).into()),
            Err(entry_number) => Err(PyTypeError::new_err(format!("Entry number {} is out of bounds; this VectorIndexTool has only has entries for {} simplices", entry_number, self.number_of_simplices()))),
        }
    }



    /// Converts a linear combination of simplices into a dense array whose length equals the number of indices in the VectorIndexTool
    /// 
    /// # Arguments
    /// 
    /// - `chain`: a Pandas DataFrame containing columns `simplex` and `coefficient`. Other columns are ignored.
    ///    Entries in the `coefficient` column should have `Frac` or `float` values.
    ///    If the same simplex appears multiple times, the coefficients are summed.
    ///   
    /// # Returns
    /// 
    /// A 1-dimensional numpy.ndarray with values representing the dense array for the column vector, where the index corresponds to the row number in the submatrix.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the input DataFrame does not contain the required columns, or if the simplices are not valid in the Vietoris-Rips complex.
    pub fn dense_array_for_dataframe<'py>(
            &self,        
            chain:       &Bound<'py, PyAny>, 
            py:          Python<'py>,       
        )
        -> PyResult<pyo3::Bound<'py, pyo3::PyAny>> 
    {
        let simplices = chain.get_item("simplex")?;
        let simplices: Vec<Vec<u16>> = simplices.extract()?;

        let coefficients = chain.get_item("coefficient")?;

        // Try to extract as floats first
        if let Ok(coeffs_float) = coefficients.extract::<Vec<f64>>() {
            let mut dense = vec![0.0f64; self.indices.len()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_float.iter()) {
                let index_number = self.index_number_for_simplex(simplex.clone())?;
                dense[index_number] += *coefficient;
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }

        // Otherwise, try to extract as fractions
        if let Ok(coeffs_frac) = coefficients.extract::<Vec<Ratio<isize>>>() {
            let mut dense = vec![Ratio::new(0, 1); self.indices.len()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_frac.iter()) {
                let index_number = self.index_number_for_simplex(simplex.clone())?;
                dense[index_number] += coefficient.clone();
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }

        // Otherwise, try to extract as integers
        if let Ok(coeffs_int) = coefficients.extract::<Vec<i64>>() {
            let mut dense = vec![0; self.indices.len()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_int.iter()) {
                let index_number = self.index_number_for_simplex(simplex.clone())?;
                dense[index_number] += coefficient.clone();
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }  

        // Otherwise, try to extract as bools
        if let Ok(coeffs_bool) = coefficients.extract::<Vec<bool>>() {
            let mut dense = vec![false; self.indices.len()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_bool.iter()) {
                let index_number = self.index_number_for_simplex(simplex.clone())?;
                dense[index_number] |= *coefficient;
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }                

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Could not extract coefficients as float, Fraction, integer, or bool types.",
        ))
    }
















    /// Converts a dense numpy array or list (with length equal to the number of rows of the submatrix) into a linear combination of simplices represented by a Pandas DataFrame
    /// 
    /// # Arguments
    /// 
    /// - `chain`: a list of `Frac` values representing the dense array for the column vector, where the index corresponds to the row number in the submatrix.
    ///   
    /// # Returns
    /// 
    /// A Pandas DataFrame with columns `simplex`, `filtration`, and `coefficient`, where `simplex` is a tuple of vertices and `coefficient` is the corresponding coefficient.
    /// 
    /// Concretely, if `dense[ i ] = v`, then row `i` of the dataframe will have `simplex = self.weighted_simplex_for_submatrix_row_number( i )` and `coefficient = v`.
    /// 
    /// Zero coefficients are ignored.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the length of the input array does not match the number of rows in the submatrix.
    pub fn dataframe_for_dense_array<'py>(
        &self,
        dense: &Bound<'py, PyAny>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let len = self.indices.len();

        // Try integers
        if let Ok(dense_vec) = dense.extract::<Vec<isize>>() {
            if dense_vec.len() != len {
                return Err(PyTypeError::new_err(format!(
                    "Length of input array {} does not match the number of rows {} in the submatrix.",
                    dense_vec.len(),
                    len
                )));
            }
            let mut output = Vec::new();
            for (index, value) in dense_vec.iter().enumerate() {
                if *value != 0 {
                    output.push((
                        self.indices.element_for_ordinal(index),
                        *value,
                    ));
                }
            }
            return output.into_dataframe_format(py);
        }

        // Try bools
        if let Ok(dense_vec) = dense.extract::<Vec<bool>>() {
            if dense_vec.len() != len {
                return Err(PyTypeError::new_err(format!(
                    "Length of input array {} does not match the number of rows {} in the submatrix.",
                    dense_vec.len(),
                    len
                )));
            }
            let mut output = Vec::new();
            for (index, value) in dense_vec.iter().enumerate() {
                if *value {
                    output.push((
                        self.indices.element_for_ordinal(index),
                        *value,
                    ));
                }
            }
            return output.into_dataframe_format(py);
        }        

        // Try fractions (Ratio<isize>)
        if let Ok(dense_vec) = dense.extract::<Vec<Ratio<isize>>>() {
            if dense_vec.len() != len {
                return Err(PyTypeError::new_err(format!(
                    "Length of input array {} does not match the number of rows {} in the submatrix.",
                    dense_vec.len(),
                    len
                )));
            }
            let mut output = Vec::new();
            for (index, value) in dense_vec.iter().enumerate() {
                if !value.is_zero() {
                    output.push((
                        self.indices.element_for_ordinal(index),
                        value.clone(),
                    ));
                }
            }
            return output.into_dataframe_format(py);
        }        


        // Try floats 
        if let Ok(dense_vec) = dense.extract::<Vec<f64>>() {
            if dense_vec.len() != len {
                return Err(PyTypeError::new_err(format!(
                    "Length of input array {} does not match the number of rows {} in the submatrix.",
                    dense_vec.len(),
                    len
                )));
            }
            let mut output = Vec::new();
            for (index, value) in dense_vec.iter().enumerate() {
                if *value != 0.0 {
                    output.push((
                        self.indices.element_for_ordinal(index),
                        *value,
                    ));
                }
            }
            return output.into_dataframe_format(py);
        }        

        Err(PyTypeError::new_err(
            "Could not extract dense array as float, Fraction, integer, or bool types.",
        ))
    }

















}












// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================










/// Stores the row and column indices of a submatrix of the boundary matrix of a Vietoris-Rips complex
/// 
/// The primary use for this object is to provide an ergonomic / readable way to map between simplices
/// and row/column numbers of a submatrix.
/// 
/// **Caution**
/// 
/// This object is not an Oracle; it stores the row and column indices in memory. If the number of
/// simplices is large, this may lead to high memory usage.
#[pyclass]
pub struct SubmatrixIndexTool{
    row_indices: BijectiveSequence<WeightedSimplex<OrderedFloat<f64>>>,
    column_indices: BijectiveSequence<WeightedSimplex<OrderedFloat<f64>>>,
    inner_vietoris_rips_complex: Arc<
                                        VietorisRipsComplex<
                                            Arc< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
                                            RingOperatorForNativeRustNumberType<RingElement>
                                        >
                                    >,
}



impl SubmatrixIndexTool {


    /// Writes a submatrix of the input matrix to a `CsMatBase` object
    /// 
    /// # Arguments
    /// 
    /// - `matrix`: a matrix oracle that provides the rows and columns of the submatrix
    /// 
    /// # Returns
    /// 
    /// A [sprs::CsMatBase] object containing the submatrix, or an error if the row or column indices are not valid in the input matrix.
    /// The `k`th row of the submatrix corresponds to the `k`th row of the submatrix indices, and similarly for columns.
    /// 
    /// **Note** The [sprs::CsMatBase] object requires and enforces that indices be strictly sorted, internally.
    /// 
    /// # Errors
    /// 
    /// Returns `Err((index_row, "row".to_owned()))` if a row index is not valid in the input matrix,
    /// and `Err((index_col, "column".to_owned()))` if a column index is not valid in the input matrix. 
    /// (only the first violating index identified is returned).
    pub fn write_submatrix_to_csmat_base< Matrix >(
            &self,
            matrix: Matrix,
        ) ->    Result<
                    CsMatBase<Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>>>,
                    // CsMatBase< FiltrationValue,      usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> >,
                    ( WeightedSimplex<OrderedFloat<f64>>, String ), 
                >
        where
            Matrix:     MatrixOracle<
                            RowIndex    =   WeightedSimplex<OrderedFloat<f64>>,
                            ColumnIndex =   WeightedSimplex<OrderedFloat<f64>>,
                            Coefficient =   Ratio<isize>,
                        >                    
    {

        // ensure that the row and column indices are valid
        for index_row in self.row_indices.vec_elements_in_order().iter() {
            if ! matrix.has_row_for_index(&index_row) {
                return Err( (index_row.clone(), "row".to_owned()) ); // skip rows that are not in the matrix
            }
        }
        for index_col in self.column_indices.vec_elements_in_order().iter() {
            if ! matrix.has_column_for_index(&index_col) {
                return Err( (index_col.clone(), "column".to_owned()) ); // skip columns that are not in the matrix
            }
        }

        let shape                           =   (self.number_of_rows(), self.number_of_columns());

        // initialize vectors for the triplets
        let mut indices_row     =   Vec::new();
        let mut indices_col     =   Vec::new();
        let mut vals     =   Vec::new();

        // generate triplets for the structural nonzeros
        for index_col in self.column_indices.vec_elements_in_order().iter().cloned() {
            for column_entry in matrix.column(&index_col) {
                let index_row = column_entry.key();
                let coefficient = column_entry.val();
                if self.row_indices.has_ordinal_for_element(&index_row) { // we screen out columns that are not in our index set
                    indices_row.push( self.row_indices.ordinal_for_element(&index_row).clone().unwrap() );
                    indices_col.push( self.column_indices.ordinal_for_element(&index_col).clone().unwrap() );                
                    vals.push( coefficient );
                }
            }
        }

        // convert triplets to a CsMatBase
        let mat                 =   TriMatBase::from_triplets(shape, indices_row, indices_col, vals);
        let mat                             =   mat.to_csr();
        return Ok( mat )
    }


}




#[pymethods]
impl SubmatrixIndexTool{ 


    fn __repr__(&self) -> String {
        format!(
            "<SubmatrixIndexTool: {} row indices, {} column indices>",
            self.row_indices.len(),
            self.column_indices.len()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "SubmatrixIndexTool\n  Number of row indices: {}\n  Number of column indices: {}",
            self.row_indices.len(),
            self.column_indices.len()
        )
    }


    /// Sets the row indices of the index tool
    /// 
    /// # Arguments
    /// 
    /// - `row_indices`: a list of list/tuples of nonnegative integers, e.g. `[[0,1,2], [1,2,3]]`.
    /// 
    ///   - Each inner list must be strictly sorted.
    ///   - The outer list can contain no duplicates.
    /// 
    /// # Errors
    /// 
    /// Returns an error if
    /// 
    /// - any of the inner lists are not strictly sorted, or
    /// - any of the simplices are not valid simplices in the Vietoris-Rips complex
    /// - the outer list contains duplicate simplices
    pub fn set_row_indices(
            &mut self, 
            row_indices: Vec<Vec<u16>>
        ) -> PyResult<()> 
    {
        let mut simplices = Vec::with_capacity(row_indices.len());

        for simplex in row_indices.iter() {
            let weighted = self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(simplex);
            match weighted {
                Ok(weighted_simplex) => {
                    simplices.push(weighted_simplex);
                },
                Err(_) => {
                    return Err(PyTypeError::new_err(format!("The input {:?} is not a valid simplex in this Vietoris-Rips complex.", &simplex)));
                },
            }            
        }

        match BijectiveSequence::from_vec( simplices ) {
            Ok(bijective_sequence) => {
                self.row_indices = bijective_sequence;
            },
            Err(e) => {
                return Err(PyTypeError::new_err( format!("User-provided indices include one or more duplicates, including {:?}.", e)));
            },
        }

        Ok(())
    }






    /// Sets the row indices of the index tool
    /// 
    /// # Arguments
    /// 
    /// - `row_indices`: a list of list/tuples of nonnegative integers, e.g. `[[0,1,2], [1,2,3]]`.
    /// 
    ///   - Each inner list must be strictly sorted.
    ///   - The outer list can contain no duplicates.
    /// 
    /// # Errors
    /// 
    /// Returns an error if
    /// 
    /// - any of the inner lists are not strictly sorted, or
    /// - any of the simplices are not valid simplices in the Vietoris-Rips complex
    /// - the outer list contains duplicate simplices
    pub fn set_column_indices(
            &mut self, 
            column_indices: Vec<Vec<u16>>
        ) -> PyResult<()> 
    {
        let mut simplices = Vec::with_capacity(column_indices.len());

        for simplex in column_indices.iter() {
            let weighted = self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(simplex);
            match weighted {
                Ok(weighted_simplex) => {
                    simplices.push(weighted_simplex);
                },
                Err(_) => {
                    return Err(PyTypeError::new_err(format!("The input {:?} is not a valid simplex in this Vietoris-Rips complex.", &simplex)));
                },
            }            
        }

        match BijectiveSequence::from_vec( simplices ) {
            Ok(bijective_sequence) => {
                self.column_indices = bijective_sequence;
            },
            Err(e) => {
                return Err(PyTypeError::new_err( format!("User-provided indices include one or more duplicates, including {:?}.", e)));
            },
        }

        Ok(())
    }    






    /// Returns the number of rows in the submatrix
    pub fn number_of_rows(&self) -> usize {
        self.row_indices.len()
    }

    /// Returns the number of columns in the submatrix
    pub fn number_of_columns(&self) -> usize {
        self.column_indices.len()
    }    

    /// Returns the collection of row indices as a dataframe
    /// 
    /// The kth row of the dataframe corresponds to the kth row of the submatrix.
    pub fn row_indices(&self, py: Python<'_>) -> PyResult<PyObject> {
        let row_indices: Vec<_> = self.row_indices.vec_elements_in_order().clone();
        row_indices.into_dataframe_format(py)
    }

    /// Returns the collection of column indices as a dataframe
    /// 
    /// The kth row of the dataframe corresponds to the kth column of the submatrix.
    pub fn column_indices(&self, py: Python<'_>) -> PyResult<PyObject> {
        let column_indices: Vec<_> = self.column_indices.vec_elements_in_order().clone();
        column_indices.into_dataframe_format(py)
    }

    /// Returns True if the given simplex is a row index in the submatrix
    pub fn contains_a_row_for_simplex(
            &self, 
            simplex: Vec<u16>
        ) -> bool
    {
        match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                self.row_indices.has_ordinal_for_element(&weighted_simplex)
            },
            Err(_) => {
                false
            },
        }
    }

    /// Returns True if the given simplex is a column index in the submatrix
    pub fn contains_a_column_for_simplex(
            &self, 
            simplex: Vec<u16>
        ) -> bool
    {
        match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                self.column_indices.has_ordinal_for_element(&weighted_simplex)
            },
            Err(_) => {
                false
            },
        }
    }    

    /// Returns the row number of the given simplex
    /// 
    /// # Arguments
    /// 
    /// - `simplex`: a strictly sorted list of vertices, e.g. `[0,1,2]`
    /// 
    /// # Returns
    /// 
    /// An integer representing the row number of the simplex in the submatrix, or an error if the Vietoris-Rips complex does not contain this simplex.
    pub fn submatrix_row_number_for_simplex(
            &self, 
            simplex: Vec<u16>
        ) -> PyResult<usize> 
    {
        match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                let row_number_opt = self.row_indices.ordinal_for_element(&weighted_simplex);
                match row_number_opt {
                    Some(row_number) => Ok(row_number),
                    None => Err(PyTypeError::new_err(format!("The SubmatrixIndexTool does not contain a row index for simplex {:?}.", &simplex))),
                }
            },
            Err(_) => {
                Err(PyTypeError::new_err(format!("User input {:?} does not represent a valid simplex in the Vietoris-Rips complex.", &simplex)))
            },
        }
    }


    /// Returns the column number of the given simplex
    /// 
    /// # Arguments
    /// 
    /// - `simplex`: a strictly sorted list of vertices, e.g. `[0,1,2]`
    /// 
    /// # Returns
    /// 
    /// An integer representing the column number of the simplex in the submatrix, or an error if the Vietoris-Rips complex does not contain this simplex.
    pub fn submatrix_column_number_for_simplex(
            &self, 
            simplex: Vec<u16>
        ) -> PyResult<usize> 
    {
        match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                let column_number_opt = self.column_indices.ordinal_for_element(&weighted_simplex);
                match column_number_opt {
                    Some(column_number) => Ok(column_number),
                    None => Err(PyTypeError::new_err(format!("The SubmatrixIndexTool does not contain a column index for simplex {:?}.", &simplex))),
                }
            },
            Err(_) => {
                Err(PyTypeError::new_err(format!("User input {:?} does not represent a valid simplex in the Vietoris-Rips complex.", &simplex)))
            },
        }
    }  


    /// Returns the simplex corresponding to the given row number
    ///
    /// # Arguments
    ///
    /// - `row_number`: an integer representing the row number in the submatrix
    ///
    /// # Returns
    ///
    /// A `WeightedSimplexPython` representing the simplex at the given row number, or an error if the row number is out of bounds.
    pub fn weighted_simplex_for_submatrix_row_number(
            &self, 
            row_number: usize
        ) -> PyResult<WeightedSimplexPython> 
    {
        match self.row_indices.element_result_for_ordinal(row_number) {
            Ok(simplex) => Ok( (&simplex).into()),
            Err(row_number) => Err(PyTypeError::new_err(format!("Row index {} is out of bounds; the submatrix has only {} rows", row_number, self.number_of_rows()))),
        }
    }


    /// Returns the simplex corresponding to the given column number
    ///
    /// # Arguments
    /// 
    /// - `column_number`: an integer representing the column number in the submatrix
    /// 
    /// # Returns
    /// 
    /// A `WeightedSimplexPython` representing the simplex at the given column number, or an error if the column number is out of bounds.
    pub fn weighted_simplex_for_submatrix_column_number(
            &self, 
            column_number: usize
        ) -> PyResult<WeightedSimplexPython> 
    {
        match self.column_indices.element_result_for_ordinal(column_number) {
            Ok(simplex) => Ok( (&simplex).into()),
            Err(column_number) => Err(PyTypeError::new_err(format!("Column index {} is out of bounds; the submatrix has only {} columns.", column_number, self.number_of_columns()))),
        }
    }



    /// Converts a linear combination of simplices into a dense array whose length equals the number of rows in the submatrix
    /// 
    /// # Arguments
    /// 
    /// - `dataframe`: a Pandas DataFrame containing columns `simplex` and `coefficient`. Other columns are ignored.
    ///    Entries in the `coefficient` column should have `Frac` values.
    ///    If the same simplex appears multiple times, the coefficients are summed.
    ///   
    /// # Returns
    /// 
    /// A 1-dimensional numpy.ndarray with `Frac` values representing the dense array for the column vector, where the index corresponds to the row number in the submatrix.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the input DataFrame does not contain the required columns, or if the simplices are not valid in the Vietoris-Rips complex.
    pub fn dense_array_for_column_vector_dataframe<'py>(
            &self,        
            dataframe:   &Bound<'py, PyAny>, 
            py:          Python<'py>,       
        )
        -> PyResult<pyo3::Bound<'py, pyo3::PyAny>> 
    {
        let simplices = dataframe.get_item("simplex")?;
        let simplices: Vec<Vec<u16>> = simplices.extract()?;

        let coefficients = dataframe.get_item("coefficient")?;


        // Try fractions (Ratio<isize>)
        if let Ok(coeffs_frac) = coefficients.extract::<Vec<Ratio<isize>>>() {
            let mut dense = vec![Ratio::new(0, 1); self.number_of_rows()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_frac.iter()) {
                let row_number = self.submatrix_row_number_for_simplex(simplex.clone())?;
                dense[row_number] += coefficient.clone();
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }

        // Try integers
        if let Ok(coeffs_int) = coefficients.extract::<Vec<isize>>() {
            let mut dense = vec![0isize; self.number_of_rows()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_int.iter()) {
                let row_number = self.submatrix_row_number_for_simplex(simplex.clone())?;
                dense[row_number] += *coefficient;
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }

        // Try bools
        if let Ok(coeffs_bool) = coefficients.extract::<Vec<bool>>() {
            let mut dense = vec![false; self.number_of_rows()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_bool.iter()) {
                let row_number = self.submatrix_row_number_for_simplex(simplex.clone())?;
                dense[row_number] |= *coefficient;
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }        

        // Try floats
        if let Ok(coeffs_float) = coefficients.extract::<Vec<f64>>() {
            let mut dense = vec![0.0f64; self.number_of_rows()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_float.iter()) {
                let row_number = self.submatrix_row_number_for_simplex(simplex.clone())?;
                dense[row_number] += *coefficient;
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }        

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Could not extract coefficients as float, Fraction, integer, or bool types.",
        ))
    }




    /// Converts a dense numpy array or list (with length equal to the number of rows of the submatrix) into a linear combination of simplices represented by a Pandas DataFrame
    /// 
    /// # Arguments
    /// 
    /// - `dense_array`: a list of values `[v0 .. vn]` corresponding to the sequence of row indices `[r0 .. rn]` contained in the :class:`SubmatrixIndexTool`.
    ///   
    /// # Returns
    /// 
    /// A Pandas DataFrame with columns `simplex`, `filtration`, and `coefficient`, where `simplex` is a tuple of vertices and `coefficient` is the corresponding coefficient.
    /// 
    /// Concretely, if `dense_array[ i ] = v`, then the output will contain a row with `simplex = self.weighted_simplex_for_submatrix_row_number( i )` and `coefficient = v`.
    /// 
    /// Zero coefficients are ignored, so the dataframe will typically have fewer rows than `len(dense_array)`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the length of the input array does not match the number of rows in the submatrix.
    pub fn column_vector_dataframe_for_dense_array<'py>(
            &self,        
            dense_array:        Vec<Bound<'py, PyAny>>, 
            py:                 Python<'py>,       
        )
        -> PyResult<PyObject> 
    {
        if dense_array.len() != self.number_of_rows() {
            return Err(PyTypeError::new_err(format!(
                "Length of input array ({}) does not match the number of rows ({}) in the submatrix.",
                dense_array.len(),
                self.number_of_rows()
            )));
        }

        let mut output = Vec::with_capacity(dense_array.len());
        
        // iterate through the dense array, checking truthiness of each entry
        for (index, value) in dense_array.into_iter().enumerate() {
            let is_nonzero = match value.is_truthy() {
                Ok(b) => b,
                Err(e) => {
                    return Err(PyTypeError::new_err(format!(
                        "Could not evaluate truthiness of value at index {}: {}. \
                        Please ensure all entries in the input vector are valid Python objects that can be evaluated as truthy (i.e., support __bool__ or __len__). \
                        This is required to identify and skip zero coefficients.",
                        index, e
                    )));
                },
            };
            if is_nonzero {
                output.push( 
                    (
                        self.row_indices.element_for_ordinal(index),
                        value,
                    )
                )
            }
        }
        output.into_dataframe_format(py)
    }










    /// Converts a linear combination of simplices into a dense array whose length equals the number of columns in the submatrix
    /// 
    /// # Arguments
    /// 
    /// - `dataframe`: a Pandas DataFrame containing columns `simplex` and `coefficient`. Other columns are ignored.
    ///    Entries in the `coefficient` column should have `Frac` values.
    ///    If the same simplex appears multiple times, the coefficients are summed.
    ///   
    /// # Returns
    /// 
    /// A 1-dimensional numpy.ndarray with `Frac` values representing the dense array for the column vector, where the index corresponds to the row number in the submatrix.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the input DataFrame does not contain the required columns, or if the simplices are not valid in the Vietoris-Rips complex.
    pub fn dense_array_for_row_vector_dataframe<'py>(
            &self,        
            dataframe:      &Bound<'py, PyAny>, 
            py:             Python<'py>,       
        )
        -> PyResult<pyo3::Bound<'py, pyo3::PyAny>> 
    {
        let simplices = dataframe.get_item("simplex")?;
        let simplices: Vec<Vec<u16>> = simplices.extract()?;

        let coefficients = dataframe.get_item("coefficient")?;

        // Try integers
        if let Ok(coeffs_int) = coefficients.extract::<Vec<isize>>() {
            let mut dense = vec![0isize; self.number_of_columns()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_int.iter()) {
                let column_number = self.submatrix_column_number_for_simplex(simplex.clone())?;
                dense[column_number] += *coefficient;
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }

        // Try bools
        if let Ok(coeffs_bool) = coefficients.extract::<Vec<bool>>() {
            let mut dense = vec![false; self.number_of_columns()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_bool.iter()) {
                let column_number = self.submatrix_column_number_for_simplex(simplex.clone())?;
                dense[column_number] |= *coefficient;
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }        

        // Try fractions (Ratio<isize>)
        if let Ok(coeffs_frac) = coefficients.extract::<Vec<Ratio<isize>>>() {
            let mut dense = vec![Ratio::new(0, 1); self.number_of_columns()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_frac.iter()) {
                let column_number = self.submatrix_column_number_for_simplex(simplex.clone())?;
                dense[column_number] += coefficient.clone();
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }        

        // Try floats
        if let Ok(coeffs_float) = coefficients.extract::<Vec<f64>>() {
            let mut dense = vec![0.0f64; self.number_of_columns()];
            for (simplex, coefficient) in simplices.iter().zip(coeffs_float.iter()) {
                let column_number = self.submatrix_column_number_for_simplex(simplex.clone())?;
                dense[column_number] += *coefficient;
            }
            let np = py.import("numpy")?;
            let dense = np.call_method1("array", (dense,))?;
            return Ok(dense);
        }        

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Could not extract coefficients as float, Fraction, integer, or bool types.",
        ))
    }



    /// Converts a dense numpy array or list (with length equal to the number of columns of the submatrix) into a linear combination of simplices represented by a Pandas DataFrame
    /// 
    /// # Arguments
    /// 
    /// - `dense_array`: a list of values `[v0 .. vn]` corresponding to the sequence of column indices `[r0 .. rn]` contained in the :class:`SubmatrixIndexTool`.
    ///   
    /// # Returns
    /// 
    /// A Pandas DataFrame with columns `simplex`, `filtration`, and `coefficient`, where `simplex` is a tuple of vertices and `coefficient` is the corresponding coefficient.
    /// 
    /// Concretely, if `dense_array[ i ] = v`, then the output will contain a row with `simplex = self.weighted_simplex_for_submatrix_column_number( i )` and `coefficient = v`.
    /// 
    /// Zero coefficients are ignored, so the dataframe will typically have fewer columns than `len(dense_array)`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the length of the input array does not match the number of columns in the submatrix.
    pub fn row_vector_dataframe_for_dense_array<'py>(
            &self,        
            dense_array:    Vec< Bound<'py, PyAny> >, 
            py:             Python<'py>,       
        )
        -> PyResult<PyObject> 
    {
        if dense_array.len() != self.number_of_columns() {
            return Err(PyTypeError::new_err(format!(
                "Length of input array {} does not match the number of columns {} in the submatrix.",
                dense_array.len(),
                self.number_of_columns()
            )));
        }

        let mut output = Vec::new();
        for (index, value) in dense_array.iter().enumerate() {
            let is_nonzero = match value.is_truthy() {
                Ok(b) => b,
                Err(e) => {
                    return Err(PyTypeError::new_err(format!(
                        "Could not evaluate truthiness of value at index {}: {}. \
                        Please ensure all entries in the input vector are valid Python objects that can be evaluated as truthy (i.e., support __bool__ or __len__). \
                        This is required to identify and skip zero coefficients.",
                        index, e
                    )));
                },
            };
            if is_nonzero {
                output.push( 
                    (
                        self.column_indices.element_for_ordinal(index),
                        value.clone(),
                    )
                )
            }
        }
        output.into_dataframe_format(py)
    }        









    /// Converts a sparse CSR matrix into a sparse Pandas DataFrame indexed by tuples of vertices
    /// 
    /// # Arguments
    /// 
    /// - `matrix`: a scipy.sparse.csr_matrix
    /// 
    /// # Returns
    /// 
    /// A Pandas DataFrame with rows and columns indexed by simplices (represented as tuples of vertices).
    /// 
    /// # Examples
    /// 
    /// See :ref:`vietoris_rips_matrices` for an example (scroll to the bottom).
    pub fn sparse_dataframe_for_csr_matrix(
        &self,
        py: Python<'_>,
        matrix: &Bound<'_, PyAny>, // Accept a scipy.sparse.csr_matrix from Python
    ) -> PyResult<PyObject> {
        // Import pandas
        let pd = py.import("pandas")?;
        // Prepare row and column labels as tuples (for better display in pandas)
        let row_labels: Vec<_> = self
            .row_indices
            .vec_elements_in_order()
            .iter()
            .map(|x| PyTuple::new(py, x.vertices()))
            .collect::<Result<Vec<_>, _>>()?;
        let col_labels: Vec<_> = self
            .column_indices
            .vec_elements_in_order()
            .iter()
            .map(|x| PyTuple::new(py, x.vertices()))
            .collect::<Result<Vec<_>, _>>()?;


        let col_labels_pylist = PyList::new(py, col_labels)?;
        let col_labels_pylist = pd.call_method1("Series", (col_labels_pylist,))?;    
        col_labels_pylist.setattr("name", "simplex")?;


        let dict = PyDict::new(py);
        dict.set_item( "index",     row_labels ).ok().unwrap();
        dict.set_item( "columns",   col_labels_pylist ).ok().unwrap();        

        // Call pandas.DataFrame.sparse.from_spmatrix
        let df = pd
            .getattr("DataFrame")?
            .getattr("sparse")?
            .call_method(
                "from_spmatrix",
                (matrix,),
                Some(&dict),
            )?;
        // df.setattr("index", df.getattr("index")?.call_method1("set_names", ("simplex",)))?;
        // let new_index = df.getattr("index")?.call_method1("set_names", ("simplex",))?;
        // df.setattr("index", new_index)?;

        Ok(df.into())
    }    








}



// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================


/// A :term:`matrix oracle` for the boundary matrix of a Vietoris-Rips complex
/// 
/// This object provides methods to compute 
/// 
/// - rows, columns, and entries of the boundary matrix
/// - the boundary of a chain
/// - the coboundary of a cochain
#[pyclass(name = "BoundaryMatrix")]
pub struct VietorisRipsBoundaryMatrixOverQ{
    inner_vietoris_rips_complex:    Arc<
                                        VietorisRipsComplex<
                                            Arc< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
                                            RingOperatorForNativeRustNumberType<RingElement>
                                        >
                                    >
}




#[pymethods]
impl VietorisRipsBoundaryMatrixOverQ{ 


    /// Construct a Vietoris-Rips complex over the field of rational numbers
    /// 
    /// # Arguments
    /// 
    /// `dissimilarity_matrix`: a sparse dissimilarity matrix stored as a Scipy sparse CSR
    ///   - missing entries will be treated as edges that never enter the filtration
    ///   - diagonal entries will be treated as vertex birth times. If `dissimilarity_matrix[i,i]` is structurally zero then vertex `i` is not included in the filtration.
    ///   - the matrix must be symmetric
    ///   - if row `i` contains any structural nonzero entry, then entry `(i,i)` must be structurally nonzero, and the smallest of all structural nonzero entries in that row
    /// 
    /// 
    /// # Returns
    /// 
    /// A `VietorisRipsBoundaryMatrixOverQ`
    /// 
    /// # Panics
    /// 
    /// Returns an error if 
    /// - `dissimilarity_matrix` is not symmetric
    /// - there exists an `i` such that entry `[i,i]` is not explicitly stored, but some other entry in row `i` *is* explicitly stored.
    ///   this is because we regard missing entries as having infinite value, rather than zero.
    /// - there exists an `i` such that entry `[i,i]` is strictly greater than some other entry in row `i`
    #[new]
    pub fn new<'py>(
            dissimilarity_matrix:       &Bound<'py, PyAny>,        
        ) 
    ->  PyResult<VietorisRipsBoundaryMatrixOverQ>
    {

        let dissimilarity_matrix = import_sparse_matrix(dissimilarity_matrix)?;
        let n_points = dissimilarity_matrix.rows();
        let dissimilarity_matrix = Arc::new( dissimilarity_matrix );                   

        // define the ring operator
        let ring_operator = FieldRationalSize::new();
        // define the chain complex
        let chain_complex_data = VietorisRipsComplex::new( 
            dissimilarity_matrix.clone(), 
            n_points, 
            ring_operator,
        );

        match chain_complex_data {
            Ok(chain_complex_data) => {
                return Ok( VietorisRipsBoundaryMatrixOverQ{ 
                    inner_vietoris_rips_complex: Arc::new( chain_complex_data ) 
                } ) 
            },
            Err(e) => {
                return Err(PyTypeError::new_err(format!("\nError constructing Vietoris-Rips complex: the dissimilarity matrix is not symmetric. \
                                                         Entry {:?} equals {:?} but entry {:?} equlas {:?}. This message is generated by by OAT.", 
                                                        e,
                                                        dissimilarity_matrix.get(e.0, e.1),
                                                        (e.1, e.0),
                                                        dissimilarity_matrix.get(e.1, e.0),
                                                    )));
            },
        }


    }    










    /// Returns the boundary of a chain
    /// 
    /// # Input
    /// 
    /// The input, `chain`, should be formatted as Pandas data frame with columns `simplex` and `coefficient`.
    /// 
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    pub fn boundary_for_chain<'py>(
            &self,        
            py:         Python<'_>,
            chain:       &Bound<'py, PyAny>,        
        )
        -> PyResult<PyObject> 
    {
        let simplices           =   chain.get_item("simplex")?;
        let simplices: Vec<Vec<u16>>           =   simplices.extract()?;
        let mut weighted_simplices = Vec::with_capacity(simplices.len());
        for simplex in simplices {
            if let Ok(filtration_value) = self.inner_vietoris_rips_complex.filtration_value_for_clique(&simplex) {
                let simplex = WeightedSimplex{ vertices: simplex, weight: filtration_value };
                weighted_simplices.push(simplex);
            } else {
                let message = format!("The input {:?} does not represent a simplex in this Vietoris-Rips complex.", &simplex);
                return Err(PyTypeError::new_err(message));
            }
        }

        let coefficients        =   chain.get_item("coefficient")?;
        let coefficients: Vec<Ratio<isize>>        =   coefficients.extract()?;
        let entries = weighted_simplices.into_iter().zip(coefficients.into_iter());
        let boundary = entries.multiply_self_as_a_column_vector_with_matrix(
                                                self.inner_vietoris_rips_complex.clone()
                                            );  
        let boundary: Vec<_> = boundary.collect();
        return boundary.into_dataframe_format(py);
    }


    /// Returns the coboundary of a cochain
    /// 
    /// # Input
    /// 
    /// The input, `cochain`, should be formatted as Pandas data frame with columns `simplex` and `coefficient`.
    /// 
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    pub fn coboundary_for_cochain<'py>(
            &self,        
            py:         Python<'_>,
            cochain:       &Bound<'py, PyAny>,        
        )
        -> PyResult<PyObject> 
    {
        let simplices           =   cochain.get_item("simplex")?;
        let simplices: Vec<Vec<u16>>           =   simplices.extract()?;
        let mut weighted_simplices = Vec::with_capacity(simplices.len());
        for simplex in simplices {
            if let Ok(filtration_value) = self.inner_vietoris_rips_complex.filtration_value_for_clique(&simplex) {
                let simplex = WeightedSimplex{ vertices: simplex, weight: filtration_value };
                weighted_simplices.push(simplex);
            } else {
                let message = format!("The input {:?} does not represent a simplex in this Vietoris-Rips complex.", &simplex);
                return Err(PyTypeError::new_err(message));
            }
        } 

        let coefficients        =   cochain.get_item("coefficient")?;
        let coefficients: Vec<Ratio<isize>>        =   coefficients.extract()?;
        let entries = weighted_simplices.into_iter().zip(coefficients.into_iter());
        let boundary = entries.multiply_self_as_a_row_vector_with_matrix(
                                                self.inner_vietoris_rips_complex.clone()
                                            );  
        let boundary: Vec<_> = boundary.collect();
        return boundary.into_dataframe_format(py);
    }    



    /// Returns the row of the matrix indexed by a given simplex
    /// 
    /// # Input
    /// 
    /// The input, `simplex`, should be a strictly sorted list or tuple of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the simplex is not contained in the Vietoris-Rips complex.
    pub fn row_for_simplex(
            &self, 
            simplex: Vec<u16>,
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                return self.inner_vietoris_rips_complex.row(&weighted_simplex).collect_vec().into_dataframe_format(py)
            },
            Err(_) => {
                return Err(PyTypeError::new_err(format!("The input {:?} does not represent a simplex in the Vietoris-Rips complex.", &simplex)))
            },
        }
    }


    /// Returns the column of the matrix indexed by a given simplex
    /// 
    /// # Input
    /// 
    /// The input, `simplex`, should be a strictly sorted list or tuple of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the simplex is not contained in the Vietoris-Rips complex.
    pub fn column_for_simplex(
            &self, 
            simplex: Vec<u16>,
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                return self.inner_vietoris_rips_complex.column(&weighted_simplex).collect_vec().into_dataframe_format(py)
            },
            Err(_) => {
                return Err(PyTypeError::new_err(format!("The input {:?} does not represent a simplex in the Vietoris-Rips complex.", &simplex)))
            },
        }
    }


    /// Returns the entry coefficient for the given row and column simplices
    /// 
    /// # Input
    /// 
    /// The inputs, `row_simplex` and `column_simplex`, should be strictly sorted lists or tuples of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Python `Fraction`
    /// 
    /// # Errors
    /// 
    /// Returns an error if one of the inputs is not a simplex contained in the Vietoris-Rips complex.
    pub fn entry_for_row_and_column(
            &self, 
            row_simplex: Vec<u16>,
            column_simplex: Vec<u16>,            
        ) -> PyResult<Ratio<isize>> 
    {
        match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&row_simplex) {
            Err(_) => {
                return Err(PyTypeError::new_err(format!("The user-provided row index {:?} does not represent a simplex in the Vietoris-Rips Complex.", &row_simplex)))
            },            
            Ok(row_weighted_simplex) => {
                match self.inner_vietoris_rips_complex.add_filtration_value_to_simplex(&column_simplex) {
                    Err(_) => {
                        return Err(PyTypeError::new_err(format!("The user-provided column index {:?} does not represent a simplex in the Vietoris-Rips Complex.", &column_simplex)))
                    },
                    Ok(column_weighted_simplex) => {
                        return Ok( 
                            self.inner_vietoris_rips_complex
                                .structural_nonzero_entry(
                                    &row_weighted_simplex,
                                    &column_weighted_simplex
                                )
                                .unwrap_or( Ratio::from_integer(0) ) 
                        )
                    },
                }
            },
        }
    }    





    /// Writes a submatrix to a Scipy CSR sparse matrix
    /// 
    /// # Arguments
    /// 
    /// - `submatrix_index_tool`: a `SubmatrixIndexTool` object containing the row and column indices of the submatrix
    /// 
    /// # Returns
    /// 
    /// A scipy.sparse.csr_matrix, or an error if the row or column indices are not valid in the input matrix.
    /// The `k`th row of the submatrix corresponds to the `k`th row of the `submatrix_index_tool`, and similarly for columns.
    /// 
    /// **Note** The coefficients of the matrix are stores as floats, because SciPy offers very limited support for
    /// rational numbers.
    pub fn write_submatrix_to_csr(
            &self, 
            py: Python<'_>,
            submatrix_index_tool: & SubmatrixIndexTool,
        ) -> PyResult<PyObject> 
    {
        match submatrix_index_tool.write_submatrix_to_csmat_base( self.inner_vietoris_rips_complex.clone() ) {
            Ok(csr_matrix) => {
                return csr_matrix.into_scipy_csr_format(py);
            },
            Err((index, kind)) => {
                return Err(PyTypeError::new_err(format!("The SubmatrixIndexTool contains {:?} index {:?}, but the desired matrix does not have a {:?} for that index.", kind, index, kind)))
            },
        }
    }

}






// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================




// #[pyclass]
// #[derive(Clone,Dissolve)]
// pub struct FloatChain{
//     pub index_tool: BijectiveSequence<WeightedSimplex<OrderedFloat<f64>>>,
//     pub coefficients: Vec<f64>,
// }


// #[pymethods]
// impl FloatChain {

//     /// Returns the number of simplices in the chain
//     pub fn len(&self) -> usize {
//         self.index_tool.len()
//     }

//     /// Returns the vector of coefficients
//     pub fn coefficients(&self) -> Vec<f64> {
//         self.coefficients.clone()
//     }

//     /// Sets the coefficients of the chain
//     pub fn set_coefficients(&mut self, coefficients: Vec<f64>) -> PyResult<()>{
//         if coefficients.len() != self.coefficients.len() {
//             return Err(PyTypeError::new_err(format!(
//                 "The number of coefficients ({}) does not match the number of simplices in the chain ({})",
//                 coefficients.len(),
//                 self.coefficients.len()
//             )));
//         }
//         self.coefficients = coefficients;
//         Ok(())
//     }


//     /// Returns a Pandas dataframe representing the chain
//     /// 
//     /// # The dataframe has columns `simplex`, `filtration`, and `coefficient`.
//     pub fn export_to_dataframe(&self, py: Python<'_>) -> PyResult<PyObject> {
//         self.into_dataframe_format(py)
//     }

// }




/// A :term:`matrix oracle` for the Laplacian matrix of a Vietoris-Rips complex
#[pyclass]
pub struct LaplacianMatrix{
    boundary_matrix:            Arc< VietorisRipsComplex<
                                          Arc< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
                                          RingOperatorForNativeRustNumberType<f64>
                                > >,
    deformation_coefficient:    f64,
}


#[pymethods]
impl LaplacianMatrix {


    // /// Returns a `FloatChain` object representing the zero chain of the given dimension
    // pub fn zero_chain( &self, dimension: isize ) -> FloatChain {
    //     let index_tool = self.boundary_matrix.cliques_in_boundary_matrix_order_fixed_dimension(dimension);
    //     let index_tool = BijectiveSequence::from_vec( index_tool ).unwrap();
    //     let num_simplices = index_tool.len();
    //     FloatChain{ index_tool, coefficients: vec![0.0; num_simplices] }
    // }

    // /// Returns a `FloatChain` object representing the zero chain of the given dimension
    // pub fn random_chain( &self, dimension: isize ) -> FloatChain {
    //     let index_tool = self.boundary_matrix.cliques_in_boundary_matrix_order_fixed_dimension(dimension);
    //     let index_tool = BijectiveSequence::from_vec( index_tool ).unwrap();
    //     let num_simplices = index_tool.len();

    //     let mut rng = thread_rng();
    //     let normal = Normal::new(0.0, 1.0).unwrap();
    //     let coefficients: Vec<f64> = (0..num_simplices)
    //         .map(|_| normal.sample(&mut rng))
    //         .collect();

    //     FloatChain{ index_tool, coefficients }
    // }    




    /// Multiplies the Laplacian with a vector
    /// 
    /// # Arguments
    /// 
    /// `chain`: a Pandas DataFrame containing columns `simplex` and `coefficient`. Other columns are ignored.
    /// 
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`.
    /// - Each entry in the `coefficient` column should be a float.
    /// 
    /// # Returns
    /// 
    /// A Pandas DataFrame with columns `simplex`, `filtration`, and `coefficient`.
    fn product_with_vector<'py>( 
            &self,        
            py:         Python<'_>,
            chain:       &Bound<'py, PyAny>, 
        )
        -> PyResult<PyObject> 
    {

        // format the chain
        // -------------------

        let simplices           =   chain.get_item("simplex")?;
        let simplices: Vec<Vec<u16>>           =   simplices.extract()?;

        // add filtration values and return errors if any simplices lie outside the Vietoris-Rips complex
        let mut weighted_simplices = Vec::with_capacity(simplices.len());        
        for simplex in simplices {
            if let Ok(filtration_value) = self.boundary_matrix.filtration_value_for_clique(&simplex) {
                let simplex = WeightedSimplex{ vertices: simplex, weight: filtration_value };
                weighted_simplices.push(simplex);
            } else {
                let message = format!("The input {:?} is not a simplex in this Vietoris-Rips complex.", &simplex);
                return Err(PyTypeError::new_err(message));
            }
        }

        let coefficients        =   chain.get_item("coefficient")?;
        let coefficients: Vec<f64>        =   coefficients.extract()?;
        let entries = weighted_simplices.into_iter().zip(coefficients.into_iter());



        // run the computation 

        let mut result = HashMap::new();        

        for (simplex, input_coefficient) in entries {

            let simplex_filtration = simplex.weight().into_inner();

            // up laplacian
            let row = self.boundary_matrix.row( &simplex );
            for row_entry in row {
                let column_index = row_entry.key();
                let column_filtration = column_index.weight().into_inner();
                let column_coefficient = row_entry.val();

                let column = self.boundary_matrix.column( & column_index );
                for column_entry in column {
                    let row_index = column_entry.key();
                    let row_filtration = row_index.weight().into_inner();
                    let row_coefficient = column_entry.val();

                    // calculate the product coefficient
                    let product_coefficient = input_coefficient * column_coefficient * row_coefficient; // the ordinary product of coefficients

                    // apply the deformation factor
                    let composite_deformation_exponent = self.deformation_coefficient * (simplex_filtration + row_filtration - 2.0 * column_filtration); // equals exp(t * (f_s + f_r - 2 * f_c))
                    let deformation_factor = composite_deformation_exponent.exp();
                    let deformed_coefficient = product_coefficient * deformation_factor;

                    // insert the result into the hashmap
                    if let Some(existing_coefficient) = result.get_mut( &row_index ) {
                        *existing_coefficient = * existing_coefficient + deformed_coefficient;
                    } else {
                        result.insert( row_index, deformed_coefficient );
                    }
                }
            }

            // down laplacian
            let column = self.boundary_matrix.column( &simplex );
            for column_entry in column {
                let row_index = column_entry.key();
                let row_filtration = row_index.weight().into_inner();
                let row_coefficient = column_entry.val();

                let row = self.boundary_matrix.row( & row_index );
                for row_entry in row {
                    let column_index = row_entry.key();
                    let column_filtration = column_index.weight().into_inner();
                    let column_coefficient = row_entry.val();

                    // calculate the product coefficient
                    let product_coefficient = input_coefficient * row_coefficient * column_coefficient; // the ordinary product of coefficients

                    // apply the deformation factor
                    let composite_deformation_exponent = self.deformation_coefficient * (2.0 * row_filtration - column_filtration - simplex_filtration); // equals exp(t * (2f_r - f_c - f_s) )
                    let deformation_factor = composite_deformation_exponent.exp();
                    let deformed_coefficient = product_coefficient * deformation_factor;

                    // insert the result into the hashmap
                    if let Some(existing_coefficient) = result.get_mut( &column_index ) {
                        *existing_coefficient = * existing_coefficient + deformed_coefficient;
                    } else {
                        result.insert( column_index, deformed_coefficient );
                    }
                }
            }
        }

        let mut result: Vec<_> = result.into_iter().collect();
        result.sort_by(|a,b| a.key().cmp(& b.key()) );
        result.into_dataframe_format( py )
    }





    /// Returns the row of the Laplacian matrix indexed by a given simplex
    /// 
    /// # Input
    /// 
    /// The input, `simplex`, should be a strictly sorted list or tuple of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    fn row_for_simplex( & self, simplex: Vec<u16>, py: Python<'_> ) -> PyResult<PyObject>
    {

        let mut result = HashMap::new();


        let simplex = self.boundary_matrix.add_filtration_value_to_simplex(&simplex)
            .map_err(|_| PyTypeError::new_err(format!("The user input {:?} is not a simplex in the Vietoris-Rips complex.", &simplex)))?;
        let simplex_filtration = simplex.weight().into_inner();


        // up laplacian
        let row = self.boundary_matrix.row( &simplex );
        for row_entry in row {
            let column_index = row_entry.key();
            let column_filtration = column_index.weight().into_inner();

            let coefficient = row_entry.val();
            for column_entry in self.boundary_matrix.column( & column_index ) {
                let row_index = column_entry.key();
                let row_filtration = row_index.weight().into_inner();
                let product_coefficient = coefficient * column_entry.val(); // the orderinary product of coeffcients

                // apply the deformation factor
                let composite_deformation_exponent = self.deformation_coefficient * (simplex_filtration + row_filtration - 2.0 * column_filtration); // equals exp(t * (f_s + f_r - 2 * f_c))
                let deformation_factor = composite_deformation_exponent.exp();
                let deformed_coefficient = product_coefficient * deformation_factor;

                // insert the result into the hashmap
                if let Some(existing_coefficient) = result.get_mut( &row_index ) {
                    *existing_coefficient = * existing_coefficient + deformed_coefficient;
                } else {
                    result.insert( row_index, deformed_coefficient );
                }
            }
        }

        // down laplacian
        let column = self.boundary_matrix.column( &simplex );
        for column_entry in column {
            let row_index = column_entry.key();
            let row_filtration = row_index.weight().into_inner();

            let row_coefficient = column_entry.val();
            for row_entry in self.boundary_matrix.row( & row_index ) {
                let column_index = row_entry.key();
                let column_filtration = column_index.weight().into_inner();
                let product_coefficient = row_coefficient * row_entry.val(); // the orderinary product of coeffcients

                // apply the deformation factor
                let composite_deformation_exponent = self.deformation_coefficient * (2.0 * row_filtration - column_filtration - simplex_filtration); // equals exp(t * (2f_r - f_c - f_s) )
                let deformation_factor = composite_deformation_exponent.exp();
                let deformed_coefficient = product_coefficient * deformation_factor;

                // insert the result into the hashmap
                if let Some(existing_coefficient) = result.get_mut( &column_index ) {
                    *existing_coefficient = * existing_coefficient + deformed_coefficient;
                } else {
                    result.insert( column_index, deformed_coefficient );
                }
            }            
        }

        let mut result: Vec<_> = result.into_iter().collect();
        result.sort_by(|a,b| a.key().cmp(& b.key()) );
        result.into_dataframe_format( py )

    }

}
























// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================







#[pyclass]
pub struct GeneralizedMatchingMatrix{
    differential_umatch:    Arc< DifferentialUmatch<
                                    // matrix
                                    Arc< VietorisRipsComplex<
                                        Arc< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
                                        RingOperatorForNativeRustNumberType<RingElement>
                                    > >,  
                                >
                            >,
    max_homology_dimension: isize,
}


#[pymethods]
impl GeneralizedMatchingMatrix {


    /// Returns an error if the simplex dimension is too high.
    fn validate_simplex_dimension( &self, simplex: Vec<u16> )  -> PyResult<()> {
        let max_homology_dimension = self.max_homology_dimension;  
        if simplex.len() as isize > max_homology_dimension + 2 {
            return Err(PyTypeError::new_err(format!(
                "Simplex {:?} has dimension {:?}, which is strictly greater than {:?}, the maximum simplex dimension for which this Umatch decomposition is valid.",
                &simplex,
                simplex.len()-1,
                max_homology_dimension
            )));
        }
        return Ok(())
    }





    /// Returns the nonzero entries of the generalized matching matrix as a dataframe
    /// 
    /// The dataframe has columns `row_simplex`, `column_simplex`, `row_filtration`, `column_filtration`, and `coefficient`.
    pub fn nonzero_entries( &self, py: Python<'_>  ) -> 
        PyResult< 
            PyObject
            // Vec< 
            //     (WeightedSimplexPython, WeightedSimplexPython, Ratio<isize>)
            // >
        >
    {
        let generalized_matching_matrix = self.differential_umatch.generalized_matching_matrix();
        let nnz = generalized_matching_matrix.number_of_structural_nonzeros();

        // fill three vectors with the triplets
        let mut row_simplices        =   Vec::with_capacity(nnz);
        let mut column_simplices     =   Vec::with_capacity(nnz);
        let mut row_filtration        =   Vec::with_capacity(nnz);
        let mut column_filtration     =   Vec::with_capacity(nnz);        
        let mut coefficients       =   Vec::with_capacity(nnz);                

        for ((row,col),val) in generalized_matching_matrix.iter_entries() {
            row_simplices.push( PyTuple::new(py, row.vertices().clone())?  );
            row_filtration.push( row.weight().into_inner() );            
            column_simplices.push( PyTuple::new(py, col.vertices().clone())? );
            column_filtration.push( col.weight().into_inner() );            
            coefficients.push( val.clone() );
        }

        // place data into a dataframe
        let dict = PyDict::new(py);
        dict.set_item( "row_simplex",        row_simplices ).ok().unwrap();
        dict.set_item( "row_filtration",        row_filtration ).ok().unwrap();        
        dict.set_item( "column_simplex",     column_simplices ).ok().unwrap();        
        dict.set_item( "column_filtration",     column_filtration ).ok().unwrap();                
        dict.set_item( "coefficient",  coefficients ).ok().unwrap();
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None).map(Into::into)        

        // let mut triplets = Vec::with_capacity( generalized_matching_matrix.number_of_structural_nonzeros() );
        // for ((row,col),val) in generalized_matching_matrix.iter_entries() {
        //     triplets.push( (row.into(), col.into(), val.clone()))            
        // }        
        // return triplets
    }



    /// Writes a submatrix to a Scipy CSR sparse matrix
    /// 
    /// # Arguments
    /// 
    /// - `submatrix_index_tool`: a `SubmatrixIndexTool` object containing the row and column indices of the submatrix
    /// 
    /// # Returns
    /// 
    /// A scipy.sparse.csr_matrix, or an error if the row or column indices are not valid in the input matrix.
    /// The `k`th row of the submatrix corresponds to the `k`th row of the `submatrix_index_tool`, and similarly for columns.
    /// 
    /// **Note** The coefficients of the matrix are stores as floats, because SciPy offers very limited support for
    /// rational numbers.
    pub fn write_submatrix_to_csr(
            &self, 
            py: Python<'_>,
            submatrix_index_tool: & SubmatrixIndexTool,
        ) -> PyResult<PyObject> 
    {
        // ensure row index dimensions do not exceed maximum
        for index_row in submatrix_index_tool.row_indices.vec_elements_in_order().iter() {
            self.validate_simplex_dimension(index_row.vertices().clone())?;
        }
        // ensure column index dimensions do not exceed maximum
        for index_col in submatrix_index_tool.column_indices.vec_elements_in_order().iter() {
            self.validate_simplex_dimension(index_col.vertices().clone())?;
        }        

        let generalized_matching_matrix = self.differential_umatch.generalized_matching_matrix();        
        match submatrix_index_tool.write_submatrix_to_csmat_base( generalized_matching_matrix ) {
            Ok(csr_matrix) => {
                return csr_matrix.into_scipy_csr_format(py);
            },
            Err((index, kind)) => {
                return Err(PyTypeError::new_err(format!("The SubmatrixIndexTool contains {:?} index {:?}, but the Vietoris-Rips complex does not contain simplex {:?}", kind, &index, &index)))
            },
        }
    }



    /// Returns the entry coefficient for the given row and column simplices
    /// 
    /// # Input
    /// 
    /// The inputs, `row_simplex` and `column_simplex`, should be strictly sorted lists or tuples of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Python `Fraction`
    /// 
    /// # Errors
    /// 
    /// Returns an error if one of the inputs is not a simplex contained in the Vietoris-Rips complex.
    pub fn entry_for_row_and_column(
            &self, 
            row_simplex: Vec<u16>,
            column_simplex: Vec<u16>,            
        ) -> PyResult<Ratio<isize>> 
    {

        // ensure row index dimensions do not exceed maximum
        self.validate_simplex_dimension(row_simplex.clone())?;
        self.validate_simplex_dimension(column_simplex.clone())?;        


        match self.differential_umatch.boundary_matrix().add_filtration_value_to_simplex(&row_simplex) {
            Err(_) => {
                return Err(PyTypeError::new_err(format!("Simplex {:?} is not a valid row index.", &row_simplex)))
            },            
            Ok(row_weighted_simplex) => {
                match self.differential_umatch.boundary_matrix().add_filtration_value_to_simplex(&column_simplex) {
                    Err(_) => {
                        return Err(PyTypeError::new_err(format!("Simplex {:?} is not a valid column index.", &column_simplex)))
                    },
                    Ok(column_weighted_simplex) => {
                        return Ok( 
                            self.differential_umatch.generalized_matching_matrix()
                                .structural_nonzero_entry(
                                    &row_weighted_simplex,
                                    &column_weighted_simplex
                                )
                                .unwrap_or( Ratio::from_integer(0) ) 
                        )
                    },
                }
            },
        }
    }      




    /// Returns a Pandas dataframe encoding the structural nonzero entries of the generalized matching matrix
    ///
    /// The dataframe has columns `coefficient`, `row_simplex`, `row_filtration`, `column_simplex`, `column_filtration`, and `filtration_difference`.
    /// The `filtration_difference` value is the difference between the `column_filtration` and `row_filtration` values.
    pub fn write_to_dataframe< 'py >( &self,  py: Python< 'py >, ) -> PyResult<PyObject> {

        // get a reference to the generalized matching matrix
        // ------------------------------------------------

        let matching = self.differential_umatch.generalized_matching_matrix();

        // preallocate vectors for the dataframe
        // ------------------------------------------------

        let n_matches = matching.number_of_structural_nonzeros();

        let mut coefficients = Vec::with_capacity( n_matches );

        let mut row_indices = Vec::with_capacity( n_matches );
        let mut column_indices = Vec::with_capacity( n_matches );

        let mut row_filtrations = Vec::with_capacity( n_matches );
        let mut column_filtrations = Vec::with_capacity( n_matches );
        let mut delta_filtrations = Vec::with_capacity( n_matches );

        // iterate over the entries of the matching matrix
        for ((row_index,column_index), coefficient) in matching.iter_entries() {
            coefficients.push(coefficient);

            let tuple = PyTuple::new(py, row_index.vertices().clone() )?;
            row_indices.push(tuple);
            let tuple = PyTuple::new(py, column_index.vertices().clone() )?;
            column_indices.push(tuple);

            row_filtrations.push( row_index.filtration().into_inner() );
            column_filtrations.push( column_index.filtration().into_inner() );
            delta_filtrations.push( column_index.filtration().into_inner() - row_index.filtration().into_inner() );
        }
        
        // create a Pandas dataframe from the vectors
        // ------------------------------------------------

        let dict = PyDict::new(py);
        dict.set_item( "coefficient", coefficients )?;   
        dict.set_item( "row_simplex", row_indices )?;        
        dict.set_item( "row_filtration", row_filtrations )?;         
        dict.set_item( "column_simplex", column_indices )?;
        dict.set_item( "column_filtration", column_filtrations )?; 
        dict.set_item( "filtration_difference", delta_filtrations )?;                 

        let pandas = py.import("pandas")?;       
        pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::into)         
    }      



    /// Returns the column index matched to a given row index, or `None` if no such column exists.
    /// 
    /// # Input
    /// 
    /// The input, `row_index`, should be a strictly sorted list or tuple of integers, e.g. `(0,1,2)`.
    ///
    /// # Output
    ///
    /// The output is a :class:`oat_python.core.vietoris_rips.WeightedSimplex` object representing the column index, or `None` if no such column exists.
    /// The column index is the (unique, if it exists) column such that `M[row_index, column_index] != 0`, where `M` is the generalized matching matrix.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the input is not a valid simplex in the Vietoris-Rips complex.
    pub fn column_index_for_row_index( &self, row_index: Vec<u16> ) -> PyResult< Option< WeightedSimplexPython > > {

        let filtration = self.differential_umatch.boundary_matrix().filtration_value_for_clique(&row_index)
            .map_err(|_| PyTypeError::new_err(format!("The user input {:?} is not a valid simplex in the Vietoris-Rips complex.", &row_index)))?;

        let row_index = WeightedSimplex{ vertices: row_index, weight: filtration };

        Ok(     self.differential_umatch
                    .generalized_matching_matrix()
                    .column_index_for_row_index(&row_index)
                    .map( |col_index| WeightedSimplexPython::from(col_index) )
        )
    }




    /// Returns the row index matched to a given column index, or `None` if no such row exists.
    /// 
    /// # Input
    /// 
    /// The input, `column_index`, should be a strictly sorted list or tuple of integers, e.g. `(0,1,2)`.
    ///
    /// # Output
    ///
    /// The output is a :class:`oat_python.core.vietoris_rips.WeightedSimplex` object representing the row index, or `None` if no such column exists.
    /// The row index is the (unique, if it exists) row such that `M[row_index, column_index] != 0`, where `M` is the generalized matching matrix.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the input is not a valid simplex in the Vietoris-Rips complex.
    pub fn row_index_for_column_index( &self, column_index: Vec<u16> ) -> PyResult< Option< WeightedSimplexPython > > {

        let filtration = self.differential_umatch.boundary_matrix().filtration_value_for_clique(&column_index)
            .map_err(|_| PyTypeError::new_err(format!("The user input {:?} is not a valid simplex in the Vietoris-Rips complex.", &column_index)))?;

        let column_index = WeightedSimplex{ vertices: column_index, weight: filtration };

        Ok(     self.differential_umatch
                    .generalized_matching_matrix()
                    .row_index_for_column_index(&column_index)
                    .map( |row_index| WeightedSimplexPython::from(row_index) )
        )
    }










}








// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================









#[pyclass]
pub struct ChangeOfBasisMatrix{
    decomposition:  Arc< DifferentialUmatch<
                            // matrix
                            Arc< VietorisRipsComplex<
                                Arc< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
                                RingOperatorForNativeRustNumberType<RingElement>
                            > >,  
                        >
                    >,
    max_homology_dimension: isize,
}

#[pymethods]
impl ChangeOfBasisMatrix{ 




    /// Returns an error if the simplex dimension is too high.
    fn validate_simplex_dimension( &self, simplex: Vec<u16> )  -> PyResult<()> {
        let max_homology_dimension = self.max_homology_dimension;  
        if simplex.len() as isize > max_homology_dimension + 2 {
            return Err(PyTypeError::new_err(format!(
                "Simplex {:?} has dimension {:?}, which is strictly greater than {:?}, the maximum simplex dimension for which this Umatch decomposition is valid.",
                &simplex,
                simplex.len()-1,
                max_homology_dimension
            )));
        }
        return Ok(())
    }





    /// Returns the product of the matrix with a column vector
    /// 
    /// # Input
    /// 
    /// The input, `chain`, should be formatted as Pandas data frame with columns `simplex` and `coefficient`.
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    pub fn product_with_column_vector<'py>(
            &self,        
            py:         Python<'_>,
            chain:       &Bound<'py, PyAny>,        
        )
        -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();
        let boundary_matrix = decomposition.boundary_matrix();
        let simplices           =   chain.get_item("simplex")?;
        let simplices: Vec<Vec<u16>>           =   simplices.extract()?;

        // ensure simplices do not exceed maximum dimension
        for simplex in simplices.iter() {
            self.validate_simplex_dimension(simplex.clone())?;
        }

        // add filtration values and return errors if any simplices like outside the Vietoris-Rips complex
        let mut weighted_simplices = Vec::with_capacity(simplices.len());        
        for simplex in simplices {
            if let Ok(filtration_value) = boundary_matrix.filtration_value_for_clique(&simplex) {
                let simplex = WeightedSimplex{ vertices: simplex, weight: filtration_value };
                weighted_simplices.push(simplex);
            } else {
                let message = format!("The input {:?} does not represent a simplex in this Vietoris-Rips complex.", &simplex);
                return Err(PyTypeError::new_err(message));
            }
        }

        let coefficients        =   chain.get_item("coefficient")?;
        let coefficients: Vec<Ratio<isize>>        =   coefficients.extract()?;
        let entries = weighted_simplices.into_iter().zip(coefficients.into_iter());
        let boundary = entries.multiply_self_as_a_column_vector_with_matrix(
                                                decomposition.differential_comb()
                                            );  
        let boundary: Vec<_> = boundary.collect();
        return boundary.into_dataframe_format(py);
    }


    /// Returns the product of the matrix with a row vector
    /// 
    /// # Input
    /// 
    /// The input, `cochain`, should be formatted as Pandas data frame with columns `simplex` and `coefficient`.
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    pub fn product_with_row_vector<'py>(
            &self,        
            py:         Python<'_>,
            cochain:       &Bound<'py, PyAny>,        
        )
        -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();        
        let boundary_matrix = decomposition.boundary_matrix();
        let simplices           =   cochain.get_item("simplex")?;
        let simplices: Vec<Vec<u16>>           =   simplices.extract()?;


        // ensure simplices do not exceed maximum dimension
        for simplex in simplices.iter() {
            self.validate_simplex_dimension(simplex.clone())?;
        }

        // add filtration values and return errors if any simplices like outside the Vietoris-Rips complex        
        let mut weighted_simplices = Vec::with_capacity(simplices.len());
        for simplex in simplices {
            if let Ok(filtration_value) = boundary_matrix.filtration_value_for_clique(&simplex) {
                let simplex = WeightedSimplex{ vertices: simplex, weight: filtration_value };
                weighted_simplices.push(simplex);
            } else {
                let message = format!("The input {:?} does not represent a simplex in this Vietoris-Rips complex.", &simplex);
                return Err(PyTypeError::new_err(message));
            }
        } 

        let coefficients        =   cochain.get_item("coefficient")?;
        let coefficients: Vec<Ratio<isize>>        =   coefficients.extract()?;
        let entries = weighted_simplices.into_iter().zip(coefficients.into_iter());
        let boundary = entries.multiply_self_as_a_row_vector_with_matrix(
                                                decomposition.differential_comb()
                                            );  
        let boundary: Vec<_> = boundary.collect();
        return boundary.into_dataframe_format(py);
    }    



    /// Returns the row of the matrix indexed by a given simplex
    /// 
    /// # Input
    /// 
    /// The input, `simplex`, should be a strictly sorted list or tuple of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the simplex is not contained in the Vietoris-Rips complex.
    pub fn row_for_simplex(
            &self, 
            simplex: Vec<u16>,
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();        

        // ensure simplex does not exceed maximum dimension
        self.validate_simplex_dimension(simplex.clone())?;

        // add filtration value and return error if simplex is not in the Vietoris-Rips complex
        match decomposition.boundary_matrix().add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                return decomposition.differential_comb().row(&weighted_simplex).collect_vec().into_dataframe_format(py)
            },
            Err(_) => {
                return Err(PyTypeError::new_err(format!("Simplex {:?} is not in the Vietoris-Rips complex.", &simplex)))
            },
        }
    }


    /// Returns the column of the matrix indexed by a given simplex
    /// 
    /// # Input
    /// 
    /// The input, `simplex`, should be a strictly sorted list or tuple of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the simplex is not contained in the Vietoris-Rips complex.
    pub fn column_for_simplex(
            &self, 
            simplex: Vec<u16>,
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();  

        // ensure simplex does not exceed maximum dimension
        self.validate_simplex_dimension(simplex.clone())?;

        // add filtration value and return error if simplex is not in the Vietoris-Rips complex              
        match decomposition.boundary_matrix().add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                return decomposition.differential_comb().column(&weighted_simplex).collect_vec().into_dataframe_format(py)
            },
            Err(_) => {
                return Err(PyTypeError::new_err(format!("Simplex {:?} is not in the Vietoris-Rips complex.", &simplex)))
            },
        }
    }









    /// Returns the entry coefficient for the given row and column simplices
    /// 
    /// # Input
    /// 
    /// The inputs, `row_simplex` and `column_simplex`, should be strictly sorted lists or tuples of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Python `Fraction`
    /// 
    /// # Errors
    /// 
    /// Returns an error if one of the inputs is not a simplex contained in the Vietoris-Rips complex.
    pub fn entry_for_row_and_column(
            &self, 
            row_simplex: Vec<u16>,
            column_simplex: Vec<u16>,            
        ) -> PyResult<Ratio<isize>> 
    {

        // ensure row index dimensions do not exceed maximum
        self.validate_simplex_dimension(row_simplex.clone())?;
        self.validate_simplex_dimension(column_simplex.clone())?;        


        match self.decomposition.boundary_matrix().add_filtration_value_to_simplex(&row_simplex) {
            Err(_) => {
                return Err(PyTypeError::new_err(format!("Simplex {:?} is not a valid row index.", &row_simplex)))
            },            
            Ok(row_weighted_simplex) => {
                match self.decomposition.boundary_matrix().add_filtration_value_to_simplex(&column_simplex) {
                    Err(_) => {
                        return Err(PyTypeError::new_err(format!("Simplex {:?} is not a valid column index.", &column_simplex)))
                    },
                    Ok(column_weighted_simplex) => {
                        return Ok( 
                            self.decomposition.differential_comb()
                                .structural_nonzero_entry(
                                    &row_weighted_simplex,
                                    &column_weighted_simplex
                                )
                                .unwrap_or( Ratio::from_integer(0) ) 
                        )
                    },
                }
            },
        }
    }      









    /// Writes a submatrix of this matrix to a Scipy CSR matrix format
    /// 
    /// # Arguments
    /// 
    /// - `submatrix_index_tool`: a `SubmatrixIndexTool` object containing the row and column indices of the submatrix
    /// 
    /// # Returns
    /// 
    /// A scipy.sparse.csr_matrix, or an error if the row or column indices are not valid in the input matrix.
    /// The `k`th row of the submatrix corresponds to the `k`th row of the `submatrix_index_tool`, and similarly for columns.
    /// 
    /// **Note** The coefficients of the matrix are stores as floats, because SciPy offers very limited support for
    /// rational numbers.
    pub fn write_submatrix_to_csr(
            &self, 
            py: Python<'_>,
            submatrix_index_tool: & SubmatrixIndexTool,
        ) -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();

        // ensure row index dimensions do not exceed maximum
        for index_row in submatrix_index_tool.row_indices.vec_elements_in_order().iter() {
            self.validate_simplex_dimension(index_row.vertices().clone())?;
        }
        // ensure column index dimensions do not exceed maximum
        for index_col in submatrix_index_tool.column_indices.vec_elements_in_order().iter() {
            self.validate_simplex_dimension(index_col.vertices().clone())?;
        }        
        
        // write the submatrix to a CSR matrix
        match submatrix_index_tool.write_submatrix_to_csmat_base( decomposition.differential_comb() ) {
            Ok(csr_matrix) => {
                return csr_matrix.into_scipy_csr_format(py);
            },
            Err((index, kind)) => {
                return Err(PyTypeError::new_err(format!("The SubmatrixIndexTool contains {:?} index {:?}, but the Vietoris-Rips complex does not contain simplex {:?}", kind, &index, &index)))
            },
        }
    }

}











// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================









#[pyclass]
pub struct ChangeOfBasisMatrixInverse{
    decomposition:   Arc< DifferentialUmatch<
                            // matrix
                            Arc< VietorisRipsComplex<
                                Arc< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
                                RingOperatorForNativeRustNumberType<RingElement>
                            > >,  
                        >
                    >,
    max_homology_dimension: isize,
}

#[pymethods]
impl ChangeOfBasisMatrixInverse{ 




    /// Returns an error if the simplex dimension is too high.
    fn validate_simplex_dimension( &self, simplex: Vec<u16> )  -> PyResult<()> {
        let max_homology_dimension = self.max_homology_dimension;  
        if simplex.len() as isize > max_homology_dimension + 2 {
            return Err(PyTypeError::new_err(format!(
                "Simplex {:?} has dimension {:?}, which is strictly greater than {:?}, the maximum simplex dimension for which this Umatch decomposition is valid.",
                &simplex,
                simplex.len()-1,
                max_homology_dimension
            )));
        }
        return Ok(())
    }





    /// Returns the product of the the matrix with a column vector
    /// 
    /// # Input
    /// 
    /// The input, `chain`, should be formatted as Pandas data frame with columns `simplex` and `coefficient`.
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    pub fn product_with_column_vector<'py>(
            &self,        
            py:         Python<'_>,
            chain:       &Bound<'py, PyAny>,        
        )
        -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();
        let boundary_matrix = decomposition.boundary_matrix();
        let simplices           =   chain.get_item("simplex")?;
        let simplices: Vec<Vec<u16>>           =   simplices.extract()?;

        // ensure simplices do not exceed maximum dimension
        for simplex in simplices.iter() {
            self.validate_simplex_dimension(simplex.clone())?;
        }

        // add filtration values and return errors if any simplices like outside the Vietoris-Rips complex
        let mut weighted_simplices = Vec::with_capacity(simplices.len());        
        for simplex in simplices {
            if let Ok(filtration_value) = boundary_matrix.filtration_value_for_clique(&simplex) {
                let simplex = WeightedSimplex{ vertices: simplex, weight: filtration_value };
                weighted_simplices.push(simplex);
            } else {
                let message = format!("The input {:?} does not represent a simplex in this Vietoris-Rips complex.", &simplex);
                return Err(PyTypeError::new_err(message));
            }
        }

        let coefficients        =   chain.get_item("coefficient")?;
        let coefficients: Vec<Ratio<isize>>        =   coefficients.extract()?;
        let entries = weighted_simplices.into_iter().zip(coefficients.into_iter());

        let product = entries.multiply_self_as_a_column_vector_with_matrix(
                                                decomposition.differential_comb_inverse()
                                            );  
        let boundary: Vec<_> = product.collect();
        return boundary.into_dataframe_format(py);
    }


    /// Returns the product of the matrix with a row vector
    /// 
    /// # Input
    /// 
    /// The input, `cochain`, should be formatted as Pandas data frame with columns `simplex` and `coefficient`.
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    pub fn product_with_row_vector<'py>(
            &self,        
            py:         Python<'_>,
            cochain:       &Bound<'py, PyAny>,        
        )
        -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();        
        let boundary_matrix = decomposition.boundary_matrix();
        let simplices           =   cochain.get_item("simplex")?;
        let simplices: Vec<Vec<u16>>           =   simplices.extract()?;


        // ensure simplices do not exceed maximum dimension
        for simplex in simplices.iter() {
            self.validate_simplex_dimension(simplex.clone())?;
        }

        // add filtration values and return errors if any simplices like outside the Vietoris-Rips complex        
        let mut weighted_simplices = Vec::with_capacity(simplices.len());
        for simplex in simplices {
            if let Ok(filtration_value) = boundary_matrix.filtration_value_for_clique(&simplex) {
                let simplex = WeightedSimplex{ vertices: simplex, weight: filtration_value };
                weighted_simplices.push(simplex);
            } else {
                let message = format!("The input {:?} does not represent a simplex in this Vietoris-Rips complex.", &simplex);
                return Err(PyTypeError::new_err(message));
            }
        } 

        let coefficients        =   cochain.get_item("coefficient")?;
        let coefficients: Vec<Ratio<isize>>        =   coefficients.extract()?;
        let entries = weighted_simplices.into_iter().zip(coefficients.into_iter());
        let product = entries.multiply_self_as_a_row_vector_with_matrix(
                                                decomposition.differential_comb_inverse()
                                            );  
        let boundary: Vec<_> = product.collect();
        return boundary.into_dataframe_format(py);
    }    



    /// Returns the row of the matrix indexed by a given simplex
    /// 
    /// # Input
    /// 
    /// The input, `simplex`, should be a strictly sorted list or tuple of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the simplex is not contained in the Vietoris-Rips complex.
    pub fn row_for_simplex(
            &self, 
            simplex: Vec<u16>,
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();        

        // ensure simplex does not exceed maximum dimension
        self.validate_simplex_dimension(simplex.clone())?;

        // add filtration value and return error if simplex is not in the Vietoris-Rips complex
        match decomposition.boundary_matrix().add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                return decomposition.differential_comb_inverse().row(&weighted_simplex).collect_vec().into_dataframe_format(py)
            },
            Err(_) => {
                return Err(PyTypeError::new_err(format!("Simplex {:?} is not in the Vietoris-Rips complex.", &simplex)))
            },
        }
    }


    /// Returns the column of the matrix indexed by a given simplex
    /// 
    /// # Input
    /// 
    /// The input, `simplex`, should be a strictly sorted list or tuple of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the simplex is not contained in the Vietoris-Rips complex.
    pub fn column_for_simplex(
            &self, 
            simplex: Vec<u16>,
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();  

        // ensure simplex does not exceed maximum dimension
        self.validate_simplex_dimension(simplex.clone())?;

        // add filtration value and return error if simplex is not in the Vietoris-Rips complex              
        match decomposition.boundary_matrix().add_filtration_value_to_simplex(&simplex) {
            Ok(weighted_simplex) => {
                return decomposition.differential_comb_inverse().column(&weighted_simplex).collect_vec().into_dataframe_format(py)
            },
            Err(_) => {
                return Err(PyTypeError::new_err(format!("Simplex {:?} is not in the Vietoris-Rips complex.", &simplex)))
            },
        }
    }









    /// Returns the entry coefficient for the given row and column simplices
    /// 
    /// # Input
    /// 
    /// The inputs, `row_simplex` and `column_simplex`, should be strictly sorted lists or tuples of integers, e.g. `(0,1,2)`.
    /// 
    /// # Output
    /// 
    /// The output is a Python `Fraction`
    /// 
    /// # Errors
    /// 
    /// Returns an error if one of the inputs is not a simplex contained in the Vietoris-Rips complex.
    pub fn entry_for_row_and_column(
            &self, 
            row_simplex: Vec<u16>,
            column_simplex: Vec<u16>,            
        ) -> PyResult<Ratio<isize>> 
    {

        // ensure row index dimensions do not exceed maximum
        self.validate_simplex_dimension(row_simplex.clone())?;
        self.validate_simplex_dimension(column_simplex.clone())?;        


        match self.decomposition.boundary_matrix().add_filtration_value_to_simplex(&row_simplex) {
            Err(_) => {
                return Err(PyTypeError::new_err(format!("Simplex {:?} is not a valid row index.", &row_simplex)))
            },            
            Ok(row_weighted_simplex) => {
                match self.decomposition.boundary_matrix().add_filtration_value_to_simplex(&column_simplex) {
                    Err(_) => {
                        return Err(PyTypeError::new_err(format!("Simplex {:?} is not a valid column index.", &column_simplex)))
                    },
                    Ok(column_weighted_simplex) => {
                        return Ok( 
                            self.decomposition.differential_comb_inverse()
                                .structural_nonzero_entry(
                                    &row_weighted_simplex,
                                    &column_weighted_simplex
                                )
                                .unwrap_or( Ratio::from_integer(0) ) 
                        )
                    },
                }
            },
        }
    }      











    /// Writes a submatrix of this matrix to a Scipy CSR matrix format
    /// 
    /// # Arguments
    /// 
    /// - `submatrix_index_tool`: a `SubmatrixIndexTool` object containing the row and column indices of the submatrix
    /// 
    /// # Returns
    /// 
    /// A scipy.sparse.csr_matrix, or an error if the row or column indices are not valid in the input matrix.
    /// The `k`th row of the submatrix corresponds to the `k`th row of the `submatrix_index_tool`, and similarly for columns.
    /// 
    /// **Note** The coefficients of the matrix are stores as floats, because SciPy offers very limited support for
    /// rational numbers.
    pub fn write_submatrix_to_csr(
            &self, 
            py: Python<'_>,
            submatrix_index_tool: & SubmatrixIndexTool,
        ) -> PyResult<PyObject> 
    {
        let decomposition = self.decomposition.clone();

        // ensure row index dimensions do not exceed maximum
        for index_row in submatrix_index_tool.row_indices.vec_elements_in_order().iter() {
            self.validate_simplex_dimension(index_row.vertices().clone())?;
        }
        // ensure column index dimensions do not exceed maximum
        for index_col in submatrix_index_tool.column_indices.vec_elements_in_order().iter() {
            self.validate_simplex_dimension(index_col.vertices().clone())?;
        }        
        
        // write the submatrix to a CSR matrix
        match submatrix_index_tool.write_submatrix_to_csmat_base( decomposition.differential_comb_inverse() ) {
            Ok(csr_matrix) => {
                return csr_matrix.into_scipy_csr_format(py);
            },
            Err((index, kind)) => {
                return Err(PyTypeError::new_err(format!("The SubmatrixIndexTool contains {:?} index {:?}, but the Vietoris-Rips complex does not contain simplex {:?}", kind, &index, &index)))
            },
        }
    }

}













// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================









/// A :term:`differential umatch decomposition` of the boundary matrix of a Vietoris-Rips complex (up to dimension `d`)
/// 
/// This object stores a minimmal number of matrix entries needed to rapidly compute any row, column, or entry of a
/// :term:`differential umatch decomposition`   :math:`(J,M,D,J)`, corresponding to matrix equation :math:`JM = DJ`,
/// where :math:`D` is 
/// 
/// - the submatrix of the boundary matrix of the Vietoris-Rips complex, containing only simplices of dimension `<= d`
/// - rows and columns of :math:`D` are ordered first by dimension, then by filtration value, then lexicographically
/// 
/// # Arguments
/// 
/// - `dissimilarity_matrix`: a sparse dissimilarity matrix stored as a Scipy sparse CSR
///   - this matrix must be symmetric
///   - missing entries will be treated as edges that never enter the filtration
///   - Diagonal entries will be treated as vertex birth times.
///   That is,  `dissimilarity_matrix[i,i]` is regarded as the filtration parameter of vertex `i`.
///   If `dissimilarity_matrix[i,i]` is structurally zero then vertex `i` is not included in the filtration.
///   If row `i` contains any structural nonzero entry, then entry `(i,i)` must be structurally nonzero, and the smallest of all structural nonzero entries in that row
/// - `max_homology_dimension`: the maximum dimension for which homology is desired
///   - Defaults to `1`.
/// - `support_fast_column_lookup`: if `True`, then the decomposition will be optimized for efficient access to the columns of `J`.
///   This also accelerates computation of cycle representatives and bounding chains in persistent homology.
/// 
///   - Defaults to `True`. However, if this argument is set to `False`, then the user can still obtain a copy optimized for column access later
///     by calling `decomposition.column_major()`, which is an out-of-place operation.
///   - Requires (modest) extra computation. The optimization for fast column lookup is a post-processing step which can be added to the regular computation.
///     Typically this post-processing step requires less time and memory than the initial decomposition.
///     If the user wishes to compute a basis of cycle representatives for persistent homology, it is typically effective to pay
///     this cost up front, because computation of cycle representatives will be much faster.
///   - Changes `J`. Performing this post-processing step will typically change the matrix `J`. In many applications,
///     it will make `J` sparser; in experiments with random point clouds, for example, `J` typically loses around two
///     thirds of its nonzero entries.  However, both versions of `J` are valid.
/// 
/// 
/// 
/// # Panics
/// 
/// Panics if 
/// 
/// - `dissimilarity_matrix` is not symmetric
/// - there exists an `i` such that entry `[i,i]` is not explicitly stored, but some other entry in row `i` *is* explicitly stored.
///   this is because we regard missing entries as having infinite value, rather than zero.
/// - there exists an `i` such that entry `[i,i]` is strictly greater than some other entry in row `i`
#[pyclass(name = "BoundaryMatrixDecomposition")]
pub struct DifferentialUmatchVietorisRipsPython{
    differential_umatch:    
                            Arc< DifferentialUmatch<
                                    // matrix
                                    Arc< VietorisRipsComplex<
                                        Arc< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
                                        RingOperatorForNativeRustNumberType<RingElement>
                                    > >,  
                                >
                            >,       
    max_homology_dimension: isize,
}




#[pymethods]
impl DifferentialUmatchVietorisRipsPython{ 


    // the documentation for the new function goes on the struct itself, not here
    #[new]
    #[pyo3(signature = (dissimilarity_matrix, max_homology_dimension=None, support_fast_column_lookup=None))]
    pub fn new<'py>(
            dissimilarity_matrix:                   &Bound<'py, PyAny>,
            max_homology_dimension:                 Option< isize >,
            support_fast_column_lookup:             Option<bool>,         
        ) 
    ->  PyResult<DifferentialUmatchVietorisRipsPython>
    {        

        let dissimilarity_matrix = import_sparse_matrix(dissimilarity_matrix)?;

        let min_homology_dimension = 0;
        let max_homology_dimension = max_homology_dimension.unwrap_or(1);

        let n_points = dissimilarity_matrix.rows();
        let dissimilarity_matrix = Arc::new( dissimilarity_matrix );                   

        // define the ring operator
        let ring_operator = FieldRationalSize::new();
        // define the chain complex
        let chain_complex_data = match VietorisRipsComplex::new( 
            dissimilarity_matrix.clone(), 
            n_points, 
            ring_operator 
        ) {
            Ok(chain_complex) => chain_complex,
            Err(e) => {
                return Err(PyTypeError::new_err(format!("\nError constructing Vietoris-Rips complex: the dissimilarity matrix is not symmetric. \
                                                         Entry {:?} equals {:?} but entry {:?} equlas {:?}. This message is generated by by OAT.", 
                                                        e,
                                                        dissimilarity_matrix.get(e.0, e.1),
                                                        (e.1, e.0),
                                                        dissimilarity_matrix.get(e.1, e.0),
                                                    )));
            },
        };
        // get a reference to the chain complex (needed in order to create certain iterators, due to lifetime bounds)
        // let chain_complex_ref = & chain_complex;   
        let chain_complex = Arc::new( chain_complex_data );
        // obtain a u-match factorization of the boundary matrix
        
        // !!! remove timer 
        let start = Instant::now();
        // !!! remove timer 

        let mut differential_umatch = DifferentialUmatch::new(
                chain_complex, 
                min_homology_dimension,   
                max_homology_dimension          
            );    

        // !!! remove timer 
        let duration = start.elapsed();
        println!("Time to factor: {:?}", duration);
        // !!! remove timer 


        // dualize if requested
        if support_fast_column_lookup.unwrap_or(true) {
            differential_umatch = differential_umatch.column_major().unwrap();
        }

        return Ok( DifferentialUmatchVietorisRipsPython{ 
            differential_umatch: Arc::new(differential_umatch), 
            max_homology_dimension 
        } ) // DifferentialUmatch { umatch, row_indices }
    }



    /// Returns a column-major version of the differential Umatch decomposition
    /// 
    /// - Out of place operation (if `self` is already column-major, then a copy of `self` is returned)
    /// - This version typically has a different differential COMB `J`. 
    /// - There is a modest computational cost to obtaining the column-major version,
    ///   but it is typically smaller than the time and memory cost of computing the original version, and
    /// - The column-major version is often enables **substantial performance gains** for computing
    ///   cycle representatives and bounding chains in persistent homology. (Indeed, any lookup operations
    ///   for the columns of the differential COMB in general).
    /// - By contrast, the original version is often more efficient for computing rows of the inverse differential COMB,
    ///   which is used for computing cocycles representatives and bounding chains in persistent cohomology.
    pub fn column_major( &self ) -> DifferentialUmatchVietorisRipsPython {

        let differential_umatch =  match self.differential_umatch.column_major() {
            Some(umatch) => Arc::new(umatch),
            None => { self.differential_umatch.clone() },
        };

        DifferentialUmatchVietorisRipsPython{
            differential_umatch,
            max_homology_dimension: self.max_homology_dimension,
        }
    }


    /// Returns `True` if this decomposition is column-major, `False` otherwise.
    pub fn is_column_major( &self ) -> bool {
        self.differential_umatch.is_column_major()
    }



    /// A helper function used for profiling.
    pub fn antitranspose_umatch_test( &self ) {

        let differential_umatch = self.differential_umatch.clone();
        let matching_matrix = self.differential_umatch.generalized_matching_matrix();

        println!(
            "number of matches in original factorization {:?}",
            matching_matrix.number_of_structural_nonzeros()
        );
        
        let start = Instant::now();

        let chain_complex = self.differential_umatch.boundary_matrix();

        let mut indices: Vec<_> = self.differential_umatch.coboundary_space_indices().iter().cloned().collect();
        indices.sort();
        println!("number of bounding indices: {:?}", indices.len());


        let mut antitranspose_umatch = Umatch::new(
            OrderAntiTranspose::new(chain_complex),
            indices.into_iter(),
        );

        let duration = start.elapsed();
        println!("Time to factor: {:?}", duration);  
        
        // COMPUTE CYCLE REPRESENTATIVES
        let start = Instant::now();  
        let mut num_classes: usize = 0;
        for index in self.differential_umatch.indices_in_homologically_valid_dimensions() {
            
            // exclude the index if it does not represent a cycle
            if self.differential_umatch.bounded_index_for(&index).is_some() {
                continue 
            }

            // if it represents a boundary, then look it up as a row of the inverse source comb in the antitranspose umatch
            if let Some(bounding_index) = self.differential_umatch.bounding_index_for(&index) {
                if index.filtration() < bounding_index.filtration() {
                    let cycle_representative = antitranspose_umatch
                        .source_comb_inverse()
                        .row_reverse(&index);
                } else {
                    continue
                }
            } else {
                // otherwise it represents an essential cycle, so we look it up as a row of the inverse target comb in the antitranspose umatch
                let cycle_representative = antitranspose_umatch
                    .target_comb_inverse()
                    .row_reverse(&index);
            }


            num_classes += 1;
        }
        let duration = start.elapsed();
        println!("Number of cycle representatives: {:?}", num_classes);
        println!("Time to retreive cycle representatives: {:?}", duration);


        // COMPUTE BOUNDING CHAINS
        let start = Instant::now();  
        num_classes = 0;
        for index in self.differential_umatch.indices_in_homologically_valid_dimensions() {
            
            // exclude the index if it does not represent a cycle
            if self.differential_umatch.bounded_index_for(&index).is_some() {
                continue 
            }

            // if it represents a boundary, then look it up as a row of the inverse source comb in the antitranspose umatch
            if let Some(bounding_index) = self.differential_umatch.bounding_index_for(&index) {
                if index.filtration() < bounding_index.filtration() {
                    let cycle_representative = antitranspose_umatch
                        .target_comb_inverse()
                        .row(&index);
                    
                    num_classes += 1;
                }
            } 
        }       
        let duration = start.elapsed();        
        println!("Number of bounding chains: {:?}", num_classes);
        println!("Time to retreive bounding chains: {:?}", duration);         
                                                 

        println!("\n\n------------------");
        println!("number of matches: {:?}", matching_matrix.number_of_structural_nonzeros());
        println!("number of simplices of dimension ≤ {:?}: {:?}", self.max_homology_dimension(),  differential_umatch.indices_in_homologically_valid_dimensions().len() );
    }






   
    /// Returns the maximum homology dimension for which this decomposition is valid
    /// 
    /// This value is determined by the user via the keyword argument `max_homology_dimension` when constructing the decomposition.
    pub fn max_homology_dimension( &self ) -> isize {
        self.max_homology_dimension
    }


    /// Returns a copy of the dissimilarity matrix used to construct the Vietoris-Rips complex
    pub fn dissimilarity_matrix( &self, py: Python<'_> ) -> PyResult< PyObject > {
        let inner_dissimilarity_matrix = (* self.differential_umatch
            .boundary_matrix()                
            .dissimilarity_matrix )
            .clone();
        return inner_dissimilarity_matrix.into_scipy_csr_format(py)
    }   



    /// Returns an error if the simplex dimension is too high.
    /// 
    /// This function checks if the dimension of the simplex is strictly greater than the maximum homology dimension for which this decomposition is valid.
    pub fn validate_simplex_dimension( &self, simplex: Vec<u16> )  -> PyResult<()> {
        let max_homology_dimension = self.max_homology_dimension();  
        if simplex.len() as isize > max_homology_dimension + 2 {
            return Err(PyTypeError::new_err(format!(
                "Simplex {:?} has dimension {:?}, which is strictly greater than {:?}, the maximum simplex dimension for which this Umatch decomposition is valid.",
                &simplex,
                simplex.len()-1,
                max_homology_dimension
            )));
        }
        return Ok(())
    }

  
    /// Returns a matrix oracle for the generalized matching matrix of the decomposition
    /// 
    /// This oracle represents the matrix `M` in the :term:`differential umatch decomposition` `(J,M,D,J)`, corresponding to the Umatch equation `JM = DJ`.
    /// 
    /// This object stores only two pieces of data: (1) the maximum homology dimension for which the Umatch decomposition is valid, and (2) a reference to the decomposition itself. **Therefore is uses almost no memory.**
    pub fn generalized_matching_matrix_oracle( &self ) -> GeneralizedMatchingMatrix {
        GeneralizedMatchingMatrix{ 
            differential_umatch: self.differential_umatch.clone(),
            max_homology_dimension: self.max_homology_dimension,
        }
    }

    /// Returns a matrix oracle for the differential COBM of the decomposition
    /// 
    /// This oracle represents the matrix `J` in the :term:`differential umatch decomposition` `(J,M,D,J)`, corresponding to the Umatch equation `JM = DJ`.
    /// 
    /// This object stores only two pieces of data: (1) the maximum homology dimension for which the Umatch decomposition is valid, and (2) a reference to the decomposition itself. **Therefore is uses almost no memory.**
    pub fn change_of_basis_matrix_oracle( &self ) -> ChangeOfBasisMatrix {
        ChangeOfBasisMatrix{ 
            decomposition: self.differential_umatch.clone(),
            max_homology_dimension: self.max_homology_dimension,
        }
    } 


    /// Returns a matrix oracle for the **inverse** differential COBM of the decomposition
    /// 
    /// This oracle represents the matrix `J^{-1}` in the :term:`differential umatch decomposition` `(J,M,D,J)`, corresponding to the Umatch equation `JM = DJ`.
    /// 
    /// This object stores only two pieces of data: (1) the maximum homology dimension for which the Umatch decomposition is valid, and (2) a reference to the decomposition itself. **Therefore is uses almost no memory.**
    pub fn change_of_basis_matrix_inverse_oracle( &self ) -> ChangeOfBasisMatrixInverse {
        ChangeOfBasisMatrixInverse{ 
            decomposition: self.differential_umatch.clone(),
            max_homology_dimension: self.max_homology_dimension,
        }
    }       



    /// Returns a matrix oracle for the boundary matrix
    /// 
    /// This oracle is lazy; it stores only a reference to the underlying weighted graph, and generates entries, rows, and columns on demand.
    pub fn boundary_matrix_oracle( &self ) -> VietorisRipsBoundaryMatrixOverQ {
        VietorisRipsBoundaryMatrixOverQ{ 
            inner_vietoris_rips_complex: self.differential_umatch.boundary_matrix().clone(),
        }
    }     


    /// Returns a copy of the underlying :class:`VietorisRipsComplex`
    /// 
    /// The returned object stores only minimal data (just a pointer to the dissimilarity
    /// matrix stored in the decomposition), so it uses very little memory. However, it can be
    /// used to generate lists of simplices, compute filtration values, and other operations.
    pub fn vietoris_rips_complex( &self ) -> VietorisRipsComplexPython {
        VietorisRipsComplexPython{ 
            inner_vietoris_rips_complex: self.differential_umatch.boundary_matrix().clone(),
        }
    }  



    /// Returns a set of row/column indices of the filtered boundary matrix, in sorted order
    /// 
    /// The indices include
    /// 
    /// - every simplex of dimension `<= d`, and 
    /// - every death (a.k.a. negative) simplex of dimension `d+1`
    /// 
    /// where `d` is the max homology dimension specified by the user when factoring the boundary matrix.
    /// 
    /// The result is a dataframe with columns `simplex` and `filtration`.
    pub fn boundary_matrix_indices_df( &self, py: Python<'_> ) 
            ->  PyResult<PyObject>        
    {
        self.boundary_matrix_indices().into_dataframe_format(py)
    }    


    /// Returns the Escolar-Hiraoka indices of a persistent cycle.
    /// 
    /// The result is a dataframe with columns `simplex` and `filtration`.
    pub fn escolar_hiraoka_indices_df( 
                &self, 
                birth_simplex:                      Vec< u16 >,
                py: Python<'_> 
            ) 
            -> PyResult<PyObject>
    {
        self.escolar_hiraoka_indices(birth_simplex).into_dataframe_format(py)
    }       


    /// Returns the indices of the zero-columns of the generalized matching matrix
    /// 
    /// These are commonly known as the "positive simplices" in persistent homology literature.
    /// 
    /// Output is a Pandas dataframe with columns `simplex` and `filtration`.
    pub fn cycle_indices(
            &self, 
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        let cycle_indices = self.differential_umatch.cycle_space_indices();
        return cycle_indices.into_dataframe_format(py);
    }


    /// Returns the indices of the nonzero columns of the generalized matching matrix in the Umatch decomposition
    /// 
    /// These are commonly known as the "negative simplices" in persistent homology literature.
    /// 
    /// Output is a Pandas dataframe with columns `simplex` and `filtration`.
    pub fn boundary_space_indices(
            &self, 
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        let boundary_space_indices = self.differential_umatch.boundary_space_indices();
        return boundary_space_indices.into_dataframe_format(py);
    }    

    /// Returns the indices of the zero rows of the generalized matching matrix
    /// 
    /// These are commonly known as the "positive simplices" in persistent **cohomology**
    /// 
    /// Output is a Pandas dataframe with columns `simplex` and `filtration`.
    pub fn cocycle_indices(
            &self, 
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        let cocycle_indices = self.differential_umatch.cocycle_space_indices();
        return cocycle_indices.into_dataframe_format(py);
    }

    /// Returns the indices of the nonzero rows of the generalized matching matrix
    /// 
    /// These are commonly known as the "negative simplices" in persistent **cohomology**
    /// 
    /// Output is a Pandas dataframe with columns `simplex` and `filtration`.
    pub fn coboundary_space_indices(
            &self, 
            py: Python<'_>,
        ) -> PyResult<PyObject> 
    {
        let coboundary_space_indices = self.differential_umatch.coboundary_space_indices();
        return coboundary_space_indices.into_dataframe_format(py);
    }    










    /// Returns the boundary matrix of the Vietoris-Rips complex,
    /// formatted as a `scipy.sparse.csr_matrix` sparse matrix
    /// 
    /// The ith row/column of this matrix corresponds to the ith simplex in `self.boundary_matrix_indices()`.
    /// See the documentation of that function for details.
    pub fn boundary_matrix_as_csr( &self, py: Python<'_> ) -> 
            PyResult<PyObject>
    {    
        self.boundary_matrix_csmat_base()
            .into_scipy_csr_format(py)
    }



    /// Returns the differential COBM `J` of a [differential Umatch decomposition](oat_rust::algebra::matrices::operations::umatch::differential)
    /// `JM = DJ`.
    /// 
    /// - `D` stands for the differential matrix (aka boundary matrix) returned by `self.boundary_matrix_as_csr()`.
    /// - The ith row/column of each matrix (`J, M, D`) corresponds to the ith simplex in `self.boundary_matrix_indices()`.
    /// - `J` is returned as a `scipy.sparse.csr_matrix` sparse matrix
    pub fn differential_comb_as_csr( &self, py: Python<'_> ) 
            -> PyResult<PyObject>
    {    
        self.differential_comb_csmat_base()
            .into_scipy_csr_format(py)
    }  


    /// Returns the *inverse* of the differential COBM `J` of a [differential Umatch decomposition](oat_rust::algebra::matrices::operations::umatch::differential)
    /// `JM = DJ`.
    /// 
    /// - `D` stands for the differential matrix (aka boundary matrix) returned by `self.boundary_matrix_as_csr()`.
    /// - The ith row/column of each matrix (`J, M, D`) corresponds to the ith simplex in `self.boundary_matrix_indices()`.
    /// - `Jinv` is returned as a `scipy.sparse.csr_matrix` sparse matrix
    pub fn differential_comb_inverse_as_csr( &self, py: Python<'_> ) 
            -> PyResult<PyObject>
    {    
        self.differential_comb_inverse_csmat_base()
            .into_scipy_csr_format(py)
    }      


    /// Returns the generalized matching matrix `M` of a [differential Umatch decomposition](oat_rust::algebra::matrices::operations::umatch::differential)
    /// `JM = DJ`.
    /// 
    /// - `D` stands for the differential matrix (aka boundary matrix) returned by `self.boundary_matrix_csmat_base()`.
    /// - The ith row/column of each matrix (`J, M, D`) corresponds to the ith simplex in `self.boundary_matrix_indices()`.
    /// - `M` is returned as a [CsMatBase] sparse matrix
    pub fn generalized_matching_matrix_as_csr( &self, py: Python<'_> ) 
            -> PyResult<PyObject>
        {
            self.generalized_matching_matrix_csmat_base()
                .into_scipy_csr_format(py)    
        }




    /// Returns a :class:`SubmatrixIndexTool` object containing ordered sequences of row and column indices
    /// 
    /// # Arguments
    /// 
    /// - `row_dimensions` (optional, defaults to `None`)
    /// 
    ///   - A list of dimensions for the row indices
    ///   - If this list has form `[d1, d2, ...]`, then the row indices will
    ///     include every simplex of dimension `d1`, `d2`. 
    ///   - If `None` is provided, then the index tool will contain row indices for
    ///     every simplex of dimension `<= d`, where `d` is the maximum homology
    ///     dimension specified by the user when creating the decomposition. 
    ///     It will **also** contain every death (a.k.a. negative) simplex of dimension `d+1`.
    /// 
    /// - `column_dimensions` (optional, defaults to `None`)
    /// 
    ///   - A list of dimensions for the column indices
    ///   - If this list has form `[d1, d2, ...]`, then the column indices will
    ///     include every simplex of dimension `d1`, `d2`. 
    ///   - If `None` is provided, then the index tool will contain column indices for
    ///     every simplex of dimension `<= d`, where `d` is the maximum homology
    ///     dimension specified by the user when creating the decomposition. 
    ///     It will **also** contain every death (a.k.a. negative) simplex of dimension `d+1`.
    /// 
    /// # Returns
    /// 
    /// A :class:`SubmatrixIndexTool` object containing the specified row and column indices.
    /// 
    /// - Indices are sorted first by dimension, then by filtration value, then lexicographically
    ///   (the same order used for the boundary matrix itself).
    /// - Duplicate dimensions are removed.
    #[pyo3(signature = (row_dimensions=None, column_dimensions=None))]
    pub fn submatrix_index_tool(
            &self, 
            row_dimensions:    Option< Vec<isize> >,
            column_dimensions: Option< Vec<isize> >,
        ) -> SubmatrixIndexTool
    {               
        let chain_complex = self.differential_umatch.boundary_matrix();      

        let row_bimap = match row_dimensions {
            Some(dimensions) => chain_complex.basis_vector_index_bimap_for_dimensions( dimensions.clone() ).ok().unwrap(),
            None => {
                let row_indices = self.boundary_matrix_indices();
                BijectiveSequence::from_iter( row_indices.into_iter() ).ok().unwrap()
            }
        };

        let column_bimap = match column_dimensions {
            Some(dimensions) => chain_complex.basis_vector_index_bimap_for_dimensions( dimensions.clone() ).ok().unwrap(),
            None => {
                let column_indices = self.boundary_matrix_indices();
                BijectiveSequence::from_iter( column_indices.into_iter() ).ok().unwrap()
            }
        };    

        SubmatrixIndexTool{
            inner_vietoris_rips_complex: chain_complex.clone(),
            row_indices: row_bimap,
            column_indices: column_bimap,
        }    
    }












    /// Returns the boundary of a chain
    /// 
    /// # Input
    /// 
    /// The input, `chain`, should be formatted as Pandas data frame with columns `simplex` and `coefficient`.
    /// 
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    pub fn boundary_for_chain<'py>(
            &self,        
            py:         Python<'_>,
            chain:       &Bound<'py, PyAny>,        
        )
        -> PyResult<PyObject> 
    {
        let boundary_matrix = self.boundary_matrix_oracle();
        return boundary_matrix.boundary_for_chain(py, chain)
    }




    /// Returns the coboundary of a cochain
    /// 
    /// # Input
    /// 
    /// The input, `cochain`, should be formatted as Pandas data frame with columns `simplex` and `coefficient`.
    /// 
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// The output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    pub fn coboundary_for_cochain<'py>(
            &self,        
            py:         Python<'_>,
            cochain:       &Bound<'py, PyAny>,        
        )
        -> PyResult<PyObject> 
    {
        let boundary_matrix = self.boundary_matrix_oracle();
        return boundary_matrix.coboundary_for_cochain(py, cochain)
    }
















    // /// Extract a barcode and a basis of cycle representatives
    // /// 
    // /// Computes the persistent homology of the filtered clique complex (ie VR complex)
    // /// with dissimilarity matrix `dissimilarity_matrix`, over the field of rational numbers.  
    // /// 
    // /// - Edges of weight `>= max_dissimilarity` are excluded.
    // /// - Homology is computed in dimensions 0 through `max_homology_dimension`, inclusive
    // /// 
    // /// Returns: `BarcodePyWeightedSimplexRational`
    // pub fn barcode( &self ) -> BarcodePyWeightedSimplexRational {
    //     // unpack the factored boundary matrix into a barcode
    //     let dim_fn = |x: &WeightedSimplex<FiltrationValue> | x.dimension() as isize;
    //     let fil_fn = |x: &WeightedSimplex<FiltrationValue> | x.filtration();    
    //     let barcode = oat_rust::algebra::chain_complexes::barcode::barcode( 
    //             self.factored.umatch(), 
    //             self.factored.row_reduction_indices().iter().cloned(), 
    //             dim_fn, 
    //             fil_fn, 
    //             true, 
    //             true
    //         );
          
    //     return BarcodePyWeightedSimplexRational::new( barcode )
    // }




    /// Solve `Dx = b` for `x`, where `D` is the boundary matrix of the Vietoris-Rips complex
    /// 
    /// # Input
    /// 
    /// The input, `b`, should be formatted as a Pandas data frame with columns `simplex` and `coefficient`.
    /// 
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    ///   - If mulitple entries in `b` have the same simplex, then they will be summed together.
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// - If a solution exists, then the output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    /// - If no solution exists, then the output is `None`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if any simplex in `b` is not contained in the Vietoris-Rips complex.
    /// Also returns an error if `b` contains a simplex of dimension greater > `max_homology_dimension + 1`,
    /// where `max_homology_dimension` is the maximum homology dimension for which this decomposition is valid
    /// (this parameter is set by the user when constructing the decomposition).
    fn solve_dx_equals_b(
            &self, 
            py: Python<'_>,
            b: &Bound<'_, PyAny>,
        ) -> PyResult<Option<PyObject>>
    {
        let boundary_matrix = self.differential_umatch.boundary_matrix();
        let simplices           =   b.get_item("simplex")?;
        let simplices: Vec<Vec<u16>>           =   simplices.extract()?;

        // ensure simplices do not exceed maximum dimension
        for simplex in simplices.iter() {
            // ensure simplex does not exceed maximum dimension
            self.validate_simplex_dimension(simplex.clone())?;
        }

        let mut weighted_simplices = Vec::with_capacity(simplices.len());
        for simplex in simplices {
            if let Ok(filtration_value) = boundary_matrix.filtration_value_for_clique(&simplex) {
                let simplex = WeightedSimplex{ vertices: simplex, weight: filtration_value };
                weighted_simplices.push(simplex);
            } else {
                let message = format!("The input {:?} does not represent a simplex in this Vietoris-Rips complex.", &simplex);
                return Err(PyTypeError::new_err(message));
            }
        }

        let coefficients        =   b.get_item("coefficient")?;
        let coefficients: Vec<Ratio<isize>>        =   coefficients.extract()?;
        let entries = weighted_simplices.into_iter().zip(coefficients.into_iter()).collect_vec();

        let umatch = self.differential_umatch.asymmetric_umatch();
        let solution    =   umatch.solve_dx_equals_b(entries);

        if let Some(solution) = solution {
            let solution = solution.into_dataframe_format(py)?;
            return Ok(Some(solution));
        } else {
            return Ok(None);
        }

    }




    /// Solve `xD = b` for `x`, where `D` is the boundary matrix of the Vietoris-Rips complex
    /// 
    /// # Input
    /// 
    /// The input, `b`, should be formatted as a Pandas data frame with columns `simplex` and `coefficient`.
    /// 
    /// - Each entry in the `coefficient` column should be a rational number, e.g. `Fraction(1,2)`
    /// - Each entry in the `simplex` column should be a strictly sorted list of vertices, e.g. `(0,1,2)`
    ///   - If mulitple entries in `b` have the same simplex, then they will be summed together.
    /// - Any other columns will be ignored.
    /// 
    /// # Output
    /// 
    /// - If a solution exists, then the output is a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`.
    /// - If no solution exists, then the output is `None`.
    /// 
    /// # Errors
    /// 
    /// Returns an error if any simplex in `b` is not contained in the Vietoris-Rips complex.
    /// Also returns an error if `b` contains a simplex of dimension greater > `max_homology_dimension + 1`,
    /// where `max_homology_dimension` is the maximum homology dimension for which this decomposition is valid
    /// (this parameter is set by the user when constructing the decomposition).
    fn solve_xd_equals_b(
            &self, 
            py: Python<'_>,
            b: &Bound<'_, PyAny>,
        ) -> PyResult<Option<PyObject>>
    {
        let boundary_matrix = self.differential_umatch.boundary_matrix();
        let simplices           =   b.get_item("simplex")?;
        let simplices: Vec<Vec<u16>>           =   simplices.extract()?;

        // ensure simplices do not exceed maximum dimension
        for simplex in simplices.iter() {
            self.validate_simplex_dimension(simplex.clone())?;
        }

        let mut weighted_simplices = Vec::with_capacity(simplices.len());
        for simplex in simplices {
            if let Ok(filtration_value) = boundary_matrix.filtration_value_for_clique(&simplex) {
                let simplex = WeightedSimplex{ vertices: simplex, weight: filtration_value };
                weighted_simplices.push(simplex);
            } else {
                let message = format!("The input {:?} does not represent a simplex in this Vietoris-Rips complex.", &simplex);
                return Err(PyTypeError::new_err(message));
            }
        }

        let coefficients        =   b.get_item("coefficient")?;
        let coefficients: Vec<Ratio<isize>>        =   coefficients.extract()?;
        let mut entries = weighted_simplices.into_iter().zip(coefficients.into_iter()).collect_vec();

        // ensure entries are sorted
        entries.sort();

        // simplify by summing the coefficients of duplicate entries
        let entries = entries.into_iter().peekable().simplify(FieldRationalSize::new());

        // solve the equation xD = b
        let umatch = self.differential_umatch.asymmetric_umatch();
        let solution    =   umatch.solve_xd_equals_b(entries);

        if let Some(solution) = solution {
            let solution = solution.into_dataframe_format(py)?;
            return Ok(Some(solution));
        } else {
            return Ok(None);
        }

    }    

    

    /// Extract a barcode and a basis of cycle representatives
    /// 
    /// Computes the persistent homology of the filtered Vietoris Rips complex 
    /// with dissimilarity matrix `dissimilarity_matrix`, over the field of rational numbers.  
    /// 
    /// - Edges of weight `>= max_dissimilarity` are excluded.
    /// - Homology is computed in dimensions 0 through `max_homology_dimension`, inclusive
    /// 
    /// Returns: a Pandas data frame with one row for each bar in the barcode. The columns are
    /// 
    /// - `dimension`: The dimension of the homology class.
    /// - `interval_length`: The length of the bar, equal to `death_filtration - birth_filtration`.
    /// - `birth_filtration`
    /// - `death_filtration`
    /// - `birth_simplex`
    /// - `death_simplex`
    /// - `cycle_representative` (optional): A cycle representative for the persistent homology class,
    ///   formatted as a data frame with columns `simplex`, `filtration`, and `coefficient`
    /// - `num_cycle_simplices` (optional): The number of nonzero coefficients in the cycle representative
    /// - `bounding_chain` (optional): A bounding chain for the cycle representative.
    ///   Concretely, if the cycle representative is `z`, then the bounding chain is a chain `x` such that `Dx = z`, where `D` is the boundary matrix.
    ///   The bounding chain is formatted as a data frame with columns `simplex`, `filtration`, and `coefficient`.
    /// - `num_bounding_chain_simplices` (optional): the number of nonzero coefficients in the bounding chain.
    pub fn persistent_homology_dataframe( 
                &self,         
                py:                             Python<'_>,
                return_cycle_representatives:   bool,
                return_bounding_chains:         bool,
        ) 
        -> PyResult<PyObject>
    {
        // unpack the factored boundary matrix into a barcode
  
        
        let barcode: 
            Barcode< 
                WeightedSimplex<OrderedFloat<f64>>, 
                ( WeightedSimplex<OrderedFloat<f64>>, Ratio<isize> ),
            >             
            = self.differential_umatch.barcode( 
                return_cycle_representatives, 
                return_bounding_chains
            );
        
        let dict = PyDict::new(py);
        dict.set_item( "id", 
            barcode.bars().iter().map(|x| x.id_number() ).collect_vec() )?;
        dict.set_item( "dimension", 
            barcode.bars().iter().map(|x| x.birth_column().dimension() ).collect_vec() )?; 
        dict.set_item( "interval_length", 
            barcode.bars().iter().map(|x| x.length_f64() ).collect_vec() )?;                       
        dict.set_item( "birth_filtration", 
            barcode.bars().iter().map(|x| x.birth_f64() ).collect_vec() )?;
        dict.set_item( "birth_simplex", 
            barcode.bars().iter().map(|x| x.birth_column().vertices().clone() ).collect_vec().into_vec_of_py_tuples(py)? )?;
        let mut death_simplices = Vec::with_capacity(barcode.len());
        for bar in barcode.bars().iter() {
            if let Some(death_column) = bar.death_column() {
                let death_simplex = PyTuple::new(py, death_column.vertices().clone() )?;
                death_simplices.push( Some(death_simplex) );
            } else {
                death_simplices.push( None );
            }
        }
        dict.set_item( "death_filtration", 
            barcode.bars().iter().map(|x| x.death_f64() ).collect_vec() )?;        
        dict.set_item( "death_simplex", death_simplices )?;        
        
        if return_cycle_representatives {

            let start = Instant::now();   // !!! remove this when done profiling

            let mut cycle_reps = Vec::with_capacity(barcode.len());
            for bar in barcode.bars().iter() {
                if let Some(cycle_representative) = bar.cycle_representative() {
                    let mut cycle_representative = cycle_representative.clone();
                    cycle_representative.sort();
                    cycle_reps.push( Some(cycle_representative.into_dataframe_format(py)?) );
                } else {
                    cycle_reps.push( None );
                }
            }

            let duration = start.elapsed(); // !!! remove this when done profiling
            println!("Time to compute cycle representatives: {:?}", duration); // !!! remove this when done profiling


            dict.set_item( "cycle_representative", cycle_reps )?;
            dict.set_item( "num_cycle_simplices", 
                barcode.bars().iter().map(|x| x.cycle_representative().as_ref().map(|x| x.len() ) ).collect_vec() )?;            
        }
        if return_bounding_chains {

            let start = Instant::now();   // !!! remove this when done profiling

            let mut bounding_chains = Vec::with_capacity(barcode.len());
            for bar in barcode.bars().iter() {
                if let Some(bounding_chain) = bar.bounding_chain() {
                    let mut bounding_chain = bounding_chain.clone();
                    bounding_chain.sort();
                    bounding_chains.push( Some(bounding_chain.clone().into_dataframe_format(py)?) );
                } else {
                    bounding_chains.push( None );
                }
            }

            let duration = start.elapsed(); // !!! remove this when done profiling
            println!("Time to compute bounding chains: {:?}", duration); // !!! remove this when done profiling            

            dict.set_item( "bounding_chain", bounding_chains)?;                    
            dict.set_item( "num_bounding_simplices", 
                barcode.bars().iter().map(|x| x.bounding_chain().as_ref().map(|x| x.len() ) ).collect_vec() )?;                                
        }
          
        let pandas = py.import("pandas")?;       
        let df: Py<PyAny> = pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::into)?;
        df.call_method( py, "set_index", ( "id", ), None)
            .map(Into::into)            
    }    


    // /// Optimize a cycle a representative with the Gurobi solver
    // /// 
    // /// As input, the function accepts the `birth_simplex` of a cycle represenative `z` for a bar `b` in persistent homology.
    // /// 
    // /// As output, it returns a cycle `c` which represents the same bar, and is as small as possible
    // /// subject to some standard conditions.  See
    // /// [Minimal Cycle Representatives in Persistent Homology Using Linear Programming: An Empirical Study With User’s Guide](https://www.frontiersin.org/articles/10.3389/frai.2021.681117/full)
    // /// for details.
    // /// 
    // /// Specifically, we employ the "edge loss" method to find a solution `x'` to the problem 
    // /// 
    // /// `minimize Cost(Ax + z)`
    // /// 
    // /// where 
    // ///
    // /// - `x` is unconstrained
    // /// - `z` is a cycle representative for the persistent homology class associated to `birth_simplex`
    // /// - `A` is a matrix whose column space equals the space of all cycles `u` such that (i) `u != z`, (ii) `u` is born no later than `z`, and (iii) `u` dies no later than `z`
    // /// - if `z` is a sum of terms of form `z_s * s`, where `s` is a simplex and `z_s` is a real number,
    // ///   then `Cost(z)` is the sum of the absolute values of the products `z_s * filtration_value(s)`.
    // /// 
    // /// Returns a data frame containing the optimal cycle, its objective value, the solution `x'` (which is labeled `difference in bounding chains`), etc.
    // /// 
    // /// This method is available when the corresponding bar in persistent homology has a finte right-endpoint.
    // #[cfg(feature = "gurobi")]
    // pub fn optimize_cycle_escolar_hiraoka< 'py >( 
    //             &self,
    //             birth_simplex:    Vec< u16 >,
    //             py: Python< 'py >,
    //         ) -> &'py PyDict { // MinimalCyclePyWeightedSimplexRational {
    //     use oat_rust::utilities::optimization::gurobi::optimize_cycle_escolar_hiraoka;
        
    //     let dim_fn = |x: &WeightedSimplex<FiltrationValue> | x.dimension() as isize;
    //     let fil_fn = |x: &WeightedSimplex<FiltrationValue> | x.filtration();    
    //     let obj_fn = |x: &WeightedSimplex<FiltrationValue> | x.filtration().into_inner();  

    //     let diam = self.factored.umatch().matrix_to_factor_ref().matrix_arc().filtration_value_for_clique(&birth_simplex).unwrap();
    //     let birth_column = WeightedSimplex{ vertices: birth_simplex, weight: diam };
    //     let optimized                   =   optimize_cycle_escolar_hiraoka(
    //                                                 self.factored.umatch(),
    //                                                 self.factored.row_reduction_indices().iter().cloned(),
    //                                                 dim_fn,
    //                                                 fil_fn,
    //                                                 obj_fn,
    //                                                 birth_column.clone(),
    //                                                 OrderOperatorTwistWeightedSimplex::new(), // we have to provide the order operator separately
    //                                             )?;   
    //     let cycle_initial                   =   optimized.cycle_initial().clone();
    //     let cycle_optimal                   =   optimized.cycle_optimal().clone();
    //     let bounding_difference              =   optimized.bounding_difference().clone();
    //     let objective_old               =   optimized.objective_initial().clone();
    //     let objective_min               =   optimized.objective_optimal().clone();

    //     let dict = PyDict::new(py);
    //     dict.set_item( "birth simplex", birth_column.vertices().clone() )?;        
    //     dict.set_item( "dimension", birth_column.vertices().len() )?;
    //     dict.set_item( "initial cycle objective value", objective_old )?;
    //     dict.set_item( "optimal cycle objective value", objective_min )?;
    //     dict.set_item( "initial cycle nnz", cycle_initial.len() )?;
    //     dict.set_item( "optimal cycle nnz", cycle_optimal.len() )?;
    //     dict.set_item( "bounding difference nnz", bounding_difference.len() )?;        
    //     dict.set_item( "initial_cycle", cycle_initial.export() )?;        
    //     dict.set_item( "optimal_cycle", cycle_optimal.export() )?;
    //     dict.set_item( "surface_between_cycles", bounding_difference.export() )?;        

    //     return dict
    // }    




    // /// Optimize a bounding chain
    // /// 
    // /// As input, the function accepts the `birth_simplex` of a cycle represenative `z` for a bar `b` in persistent homology.
    // /// 
    // /// As output, it returns a cycle `c` which represents the same bar, and is as small as possible
    // /// subject to some standard conditions.  See
    // /// [Minimal Cycle Representatives in Persistent Homology Using Linear Programming: An Empirical Study With User’s Guide](https://www.frontiersin.org/articles/10.3389/frai.2021.681117/full)
    // /// for details.
    // /// 
    // /// Specifically, we employ the "edge loss" method to find a solution `x'` to the problem 
    // /// 
    // /// `minimize Cost(Ax + z)`
    // /// 
    // /// where 
    // ///
    // /// - `x` is unconstrained
    // /// - `z` is a cycle representative for the persistent homology class associated to `birth_simplex`
    // /// - `A` is a matrix whose column space equals the space of all cycles `u` such that (i) `u != z`, (ii) `u` is born no later than `z`, and (iii) `u` dies no later than `z`
    // /// - if `z` is a sum of terms of form `z_s * s`, where `s` is a simplex and `z_s` is a real number,
    // ///   then `Cost(z)` is the sum of the absolute values of the products `z_s * filtration_value(s)`.
    // /// 
    // /// Returns a data frame containing the optimal cycle, its objective value, the solution `x'` (which is labeled `difference in bounding chains`), etc.
    // /// 
    // /// This method is available when the corresponding bar in persistent homology has a finte right-endpoint.
    // pub fn optimize_cycle< 'py >( 
    //             &self,
    //             birth_simplex:    Vec< u16 >,
    //             py: Python< 'py >,
    //         ) -> &'py PyDict { // MinimalCyclePyWeightedSimplexRational {

    //     // inputs
    //     let matching                  =   self.factored.umatch().generalized_matching_matrix_ref();        
    //     let order_operator                  =   self.factored.umatch().order_operator_for_row_entries_reverse();
        
    //     // matrix a, vector c, and the dimension function
    //     let dim_fn = |x: &WeightedSimplex<FiltrationValue> | x.dimension() as isize;
    //     let obj_fn = |x: &WeightedSimplex<FiltrationValue> | x.filtration().into_inner(); 
    //     let a = |k: &WeightedSimplex<FiltrationValue>| self.factored.jordan_basis_vector( &k ); 
             
    //     // column b
    //     let diam = self.factored.umatch().matrix_to_factor_ref().matrix_arc().filtration_value_for_clique(&birth_simplex).unwrap();
    //     let birth_column = WeightedSimplex{ vertices: birth_simplex, weight: diam };
    //     let b = self.factored.jordan_basis_vector( birth_column.clone() );  

    //     // column indices of a
    //     let column_indices  =   self.factored.escolar_hiraoka_indices( birth_column.clone(), dim_fn );

    //     // solve
    //     let optimized = oat_rust::utilities::optimization::minimize_l1::minimize_l1(a, b, obj_fn, column_indices).unwrap();

    //     // formatting
    //     let to_ratio = |x: f64| -> Ratio<isize> { Ratio::<isize>::approximate_float(x).unwrap() };
    //     let format_chain = |x: Vec<_>| {
    //         let mut r = x
    //             .into_iter()
    //             .map(|(k,v): (WeightedSimplex<_>,f64) | (k,to_ratio(v)))
    //             .collect_vec();
    //         // r.sort_by( |&(k,v), &(l,u)| order_operator.judge_cmp(&l, &k) );
    //         r.sort_by( |a,b| order_operator.judge_cmp(a, b) );
    //         r
    //     };
        
    //     // optimal solution data
    //     let x =     format_chain( optimized.x().clone() );    
    //     println!("{:?}", &x);    
    //     let cycle_optimal =     format_chain( optimized.y().clone() );
    //     let cycle_initial =     optimized.b().clone();        


    //     // triangles involved
    //     let bounding_difference             =   
    //         x.iter().cloned()
    //         .filter( |x| matching.has_a_match_for_row_index( &x.0) ) // only take entries for boundaries
    //         .map(|(k,v)| (matching.column_index_for_row_index( &k ).clone().unwrap(),v) )
    //         .multiply_self_as_a_column_vector_with_matrix_and_return_entries_in_reverse_order( self.factored.differential_comb() )
    //         .collect_vec();

    //     // essential cycles involved
    //     let essential_difference            =   
    //         x.iter().cloned()
    //         .filter( |x| matching.lacks_a_match_for_column_index( &x.0 ) ) // only take entries for boundaries
    //         .multiply_self_as_a_column_vector_with_matrix_and_return_entries_in_reverse_order( self.factored.differential_comb() )
    //         .collect_vec();       

    //     let objective_old               =   optimized.cost_b().clone();
    //     let objective_min               =   optimized.cost_y().clone();

    //     let dict = PyDict::new(py);
    //     dict.set_item( "birth simplex", birth_column.vertices().clone() )?;        
    //     dict.set_item( "dimension", birth_column.vertices().len() as isize - 1 )?;
    //     dict.set_item( "initial cycle objective value", objective_old )?;
    //     dict.set_item( "optimal cycle objective value", objective_min )?;
    //     dict.set_item( "initial cycle nnz", cycle_initial.len() )?;
    //     dict.set_item( "optimal cycle nnz", cycle_optimal.len() )?;
    //     dict.set_item( "initial_cycle", cycle_initial.export() )?;        
    //     dict.set_item( "optimal_cycle", cycle_optimal.export() )?;
    //     dict.set_item( "difference in bounding chains nnz", bounding_difference.len() )?;         
    //     dict.set_item( "surface_between_cycles", bounding_difference.export() )?;   
    //     dict.set_item( "difference in essential cycles nnz", essential_difference.len() )?;                                            
    //     dict.set_item( "difference_in_essential cycles", essential_difference.export() )?;

    //     return dict
    // }  



















    /// Optimize a cycle representative
    /// 
    /// This is a method to find tight cycle reprepresentaives in (persistent) homology. The output is typically a cycle with fewer simplices.
    /// 
    /// Mathematically, it uses the "edge loss" method to find a solution `x'` to the problem 
    /// 
    /// `minimize Cost(Ax + z)`
    /// 
    /// where 
    ///
    /// - `x` is unconstrained
    /// - `z` is a cycle representative for a (persistent) homology class associated to a given `birth_simplex`
    /// - `A` is a matrix composed of a subset of columns of the [differential COBM](oat_rust::algebra::matrices::operations::umatch::differential) `J` of a [differential Umatch decomposition](oat_rust::algebra::matrices::operations::umatch::differential)
    /// - `Cost(z)` is the sum of the absolute values of the products `z_s * filtration_value(s)`.
    /// 
    /// Parameters
    /// ----------------------------
    /// 
    /// - The `birth_simplex` of a cycle represenative `z` for a bar `b` in persistent homology.
    /// - The `problem_type` type for the problem. The optimization procedure works by adding linear combinations of column vectors from the Jordan basis matrix computed in the factorization. This argument controls which columns are available for the combination.
    /// 
    ///   - (default) **"preserve PH basis"** adds cycles which appear strictly before `birth_simplex`
    ///     in the standard ordering on filtered simplex (first by filtration, then breaking ties by
    ///     lexicographic order on simplices) and die no later than the persistent homology class represented by `birth_simplex`.  **Note** this is
    ///     almost the same as the problem described in [Escolar and Hiraoka, Optimal Cycles for 
    ///     Persistent Homology Via Linear Programming](https://link.springer.com/chapter/10.1007/978-4-431-55420-2_5)
    ///     except that we can include essential cycles, if `birth_simplex` represents an essential class.
    /// 
    ///   - **"preserve homology class"** includes every column of `J` that lies in the image of the boundary
    ///     operator at or before the filtration value of `birth_simplex`
    /// 
    /// Returns
    /// ----------------------------
    /// 
    /// A pandas dataframe containing the following vectors
    /// 
    /// - ``initial_cycle``: the initial cycle representative returned by the ``decomposition``
    /// - ``optimal_cycle``: the optimal cycle representative. This cycle has form 
    /// 
    ///   .. math::
    ///     o = z + e + Dc
    /// 
    ///   where
    ///   - :math:`o` is the optimal cycle,
    ///   - :math:`z` is the initial cycle,
    ///   - :math:`c` is a chain and :math:`Dc` is the boundary of :math:`c`,
    ///   - :math:`e` is a chain in the space spanned by essential cycles (that is, cycles which never become boundaries).
    /// 
    ///     - Typically :math:`e` is zero, so :math:`o = z + Dc`. In fact this is will *always* true if :math:`z` is non-essential, i.e. if :math:`z` represents
    ///       persistent homology class with a finite death time. If this is true, then :math:`z` and `o` will eventually become homologous, since
    ///       they differ by a boundary. However, :math:`c` may have a birth time strictly later than :math:`z`, so :math:`z` and `o` may not be
    ///       homologous at the birth time of :math:`z`.
    /// 
    /// - ``surface_between_cycles`` is a chain :math:`c`. You can think of this chain, informally,
    ///   as a surface whose boundary is the difference between the initial and optimal cycles. In particular, if :math:`e=0` then :math:`0 = z + Dc`.
    /// - ``difference_in_essential_cycles``: is the chain :math:`e` in the decomposition above.
    /// 
    /// 
    /// Examples
    /// ----------------------------
    /// 
    /// See :ref:`vietoris_rips_dragon` for an example.
    /// 
    /// 
    /// Related
    /// ----------------------------
    /// 
    /// See
    /// 
    /// - [Escolar and Hiraoka, Optimal Cycles for Persistent Homology Via Linear Programming](https://link.springer.com/chapter/10.1007/978-4-431-55420-2_5)
    /// - [Obayashi, Tightest representative cycle of a generator in persistent homology](https://epubs.siam.org/doi/10.1137/17M1159439)
    /// - [Minimal Cycle Representatives in Persistent Homology Using Linear Programming: An Empirical Study With User’s Guide](https://www.frontiersin.org/articles/10.3389/frai.2021.681117/full)
    /// 
    #[pyo3(signature = (birth_simplex, problem_type=None, verbose=true))]
    pub fn optimize_cycle< 'py >( 
                &self,
                birth_simplex:                      Vec< u16 >,
                problem_type:                       Option< &str >,
                verbose:                            bool,
                py: Python< 'py >,
            ) -> PyResult<PyObject> { // MinimalCyclePyWeightedSimplexRational {

        // inputs
        let matching                  =   self.differential_umatch.generalized_matching_matrix();        
        let ring_operator             =   self.differential_umatch.ring_operator();
        
        // matrix a, vector c, and the dimension function
        let dim_fn = |x: &WeightedSimplex<FiltrationValue> | x.dimension() as isize;
        let obj_fn = |x: &WeightedSimplex<FiltrationValue> | x.filtration().into_inner(); 
        let a = |k: &WeightedSimplex<FiltrationValue>| self.differential_umatch.differential_comb().column( &k ); 
             
        // column b
        let diam = self.differential_umatch.boundary_matrix().filtration_value_for_clique(&birth_simplex).unwrap();
        let birth_column = WeightedSimplex{ vertices: birth_simplex.clone(), weight: diam };
        let dimension = birth_column.dimension();
        let b = self.differential_umatch.differential_comb().column( &birth_column );

        let column_indices = match problem_type.unwrap_or("preserve PH basis") {
            "preserve homology class"    =>  {
                self.differential_umatch
                    .boundary_space_indices() // indices of all boundary vectors in the differential COBM
                    .into_iter()
                    .filter(
                        |x| 
                        ( x.dimension()==dimension )
                        && 
                        ( x.filtration() <= diam )
                    ) // of appropriate dimension    
                    .collect_vec()            
            }     
            "preserve PH basis"    =>  {
                self.differential_umatch
                    .escolar_hiraoka_indices( & birth_column, dim_fn, ) // indices of all boundary vectors in the differential COBM
            }
            _ => {
                return Err( PyErr::new::<pyo3::exceptions::PyValueError, _>( "The `problem_type` argument must be one of the following: `preserve homology class`, `preserve homology basis (once)`, `preserve PH basis (once)`, or `preserve PH basis`.\nThis error message is generated by OAT." ) );
            }                              
        };



        // function to turn floats into rationals
        let to_ratio = |x: f64| -> Ratio<isize> { 
            let frac    =   Ratio::<isize>::approximate_float(x);
            if frac == None { println!("unconvertible float: {:?}", x); }
            frac.unwrap()
        };

        // function to change R-linear combination to a Q-linear combination
        let order_operator = self.differential_umatch.boundary_matrix().order_operator_for_column_entries();
        let format_chain = |x: Vec<_>| {
            let mut r = x
                .into_iter()
                .map(|(k,v): (WeightedSimplex<_>,f64) | (k,to_ratio(v)))
                .collect_vec();
            r.sort_by( |a,b| 
                order_operator.judge_cmp(a, b) 
            );
            r
        };


        // solve
        let optimized = oat_rust::utilities::optimization::minimize_l1_try_gurobi(
            a, 
            b, 
            obj_fn, 
            column_indices, 
            verbose
        ).unwrap();         
        
        // optimal solution data
        let x                           =     format_chain( optimized.x().clone() );       
        let cycle_optimal               =     format_chain( optimized.y().clone() );
        let cycle_initial               =     optimized.b().clone();        



        // triangles involved
        let mut surface_between_cycles = Vec::with_capacity(x.len()); // first compute a vector v such that DJv = Jx (modulo essential cycles)
        for (index,coefficient) in x.iter().cloned() {
            if let Some( (column_index, matching_coefficient) ) = matching.row( &index ).next() { // only take entries for boundaries                
                let corrected_coefficient = ring_operator.divide(coefficient, matching_coefficient);
                surface_between_cycles.push( (column_index,  corrected_coefficient) );
            }
        }
        let surface_between_cycles = surface_between_cycles
                .into_iter()
                .multiply_self_as_a_column_vector_with_matrix( 
                    self.differential_umatch.differential_comb() 
                )
                .collect_vec();

        // essential cycles involved
        let essential_difference    =   
            x.iter().cloned()
            .filter( |x| matching.lacks_a_match_for_row_index( &x.0 ) ) // only take entries for boundaries
            .multiply_self_as_a_column_vector_with_matrix( 
                self.differential_umatch.differential_comb() 
            )
            .collect_vec();      

        // //  CHECK THE RESULTS
        // //  --------------------
        // //
        // //  * COMPUTE (Ax + z) - y
        // //  * ENSURE ALL VECTORS ARE SORTED

        // let ring_operator   =   self.decomposition.umatch().ring_operator();      

        // // We place all iterators in wrappers that check that the results are sorted
        // let y   =   RequireStrictAscentWithPanic::new( 
        //                     cycle_optimal.iter().cloned(),  // sorted in reverse
        //                     order_operator,                 // judges order in reverse
        //                 );
        

        // let z   =   RequireStrictAscentWithPanic::new( 
        //                     cycle_initial.iter().cloned(),  // sorted in reverse
        //                     order_operator,                 // judges order in reverse
        //                 );                                           
            
        // // the portion of Ax that comes from essential cycles;  we have go through this more complicated construction, rather than simply multiplying by the jordan basis matrix, because we've changed basis for the bounding difference chain
        // let ax0 =   RequireStrictAscentWithPanic::new( 
        //                     essential_difference.iter().cloned(),   // sorted in reverse
        //                     order_operator,                         // judges order in reverse
        //                 );                  

        // // the portion of Ax that comes from non-essential cycles;  we have go through this more complicated construction, rather than simply multiplying by the jordan basis matrix, because we've changed basis for the bounding difference chain
        // let ax1
        //     =   RequireStrictAscentWithPanic::new( 
        //             bounding_difference
        //                 .iter()
        //                 .cloned()
        //                 .multiply_self_as_a_column_vector_with_matrix_and_return_entries_in_reverse_order(self.decomposition.umatch().matrix_to_factor_ref()),  // sorted in reverse
        //             order_operator,                 // judges order in reverse
        //         );  


        // let ax_plus_z_minus_y
        //     =   RequireStrictAscentWithPanic::new( 
        //             ax0.peekable()
        //                 .add(
        //                         ax1.peekable(),
        //                         ring_operator,
        //                         order_operator,
        //                     )
        //                 .peekable()
        //                 .add(
        //                         z.into_iter().peekable(),
        //                         ring_operator,
        //                         order_operator,
        //                     )
        //                 .peekable()
        //                 .subtract(
        //                         y.into_iter().peekable(),
        //                         ring_operator,
        //                         order_operator,
        //                     ),
        //             order_operator,                 
        //         )
        //         .collect_vec();      

        let dict = PyDict::new(py);

        // row labels
        dict.set_item(
            "variable", 
            vec![
                "initial cycle", 
                "optimal_cycle", 
                "x",
                "surface_between_cycles",
                "difference_in_essential_cycles",
                "time_to_formulate_the_problem",
                "time_to_solve_the_problem",
            ]
        )?;

        // objective costs
        dict.set_item(
            "cost", 
            vec![ 
                Some(optimized.cost_b().clone()), 
                Some(optimized.cost_y().clone()), 
                None, 
                None,
                None,
                Some(optimized.construction_time),
                Some(optimized.solve_time),
            ] 
        )?; 

        // number of nonzero entries per vector       
        dict.set_item(
            "num_nonzero_coefficients", 
            vec![ 
                Some(cycle_initial.len()), 
                Some(cycle_optimal.len()), 
                Some(x.len()),
                Some(surface_between_cycles.len()),
                Some(essential_difference.len()),
                None,
                None,
            ] 
        )?;

        // vectors
        dict.set_item(
            "chain", 
            vec![ 
                Some(cycle_initial.into_dataframe_format(py).ok()), 
                Some(cycle_optimal.into_dataframe_format(py).ok()), 
                Some(x.into_dataframe_format(py).ok()),
                Some(surface_between_cycles.into_dataframe_format(py).ok()),
                Some(essential_difference.into_dataframe_format(py).ok()),
                None,
                None,
                ] 
        )?;   

        let pandas = py.import("pandas")?;       
        let dict = pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::< Py<PyAny> >::into)?;
        let kwarg = vec![("inplace", true)].into_py_dict(py)?;        
        dict.call_method( py, "set_index", ( "variable", ), Some(&kwarg))?;        

        return Ok(dict)
    }        



    /// Finds a chain `c` whose boundary is the cycle representative of `birth_simplex`, such that
    /// the l1 norm of `c` is as small as possible
    #[pyo3(signature = (birth_simplex, verbose=true))]
    pub fn optimize_bounding_chain< 'py >( 
                &self,
                birth_simplex:                      Vec< u16 >,
                verbose:                            bool,
                py: Python< 'py >,
            ) -> PyResult<PyObject> { // MinimalCyclePyWeightedSimplexRational {

        // inputs
        let boundary_matrix                   =   self.differential_umatch.boundary_matrix();                        
        let matching                  =   self.differential_umatch.generalized_matching_matrix();    
        let order_operator                  =   self.differential_umatch.boundary_matrix().order_operator_for_row_entries_reverse();
        
        // matrix a, vector c, and the dimension function
        let dim_fn = |x: &WeightedSimplex<FiltrationValue> | x.dimension() as isize;
        let obj_fn = |x: &WeightedSimplex<FiltrationValue> | 1.0; // x.filtration().into_inner(); 
        let a = |k: &WeightedSimplex<FiltrationValue>| self.differential_umatch.differential_comb().column_reverse( &k ); 
             
        // column b
        let diam = self.differential_umatch.boundary_matrix().filtration_value_for_clique(&birth_simplex).unwrap();
        let birth_column = WeightedSimplex{ vertices: birth_simplex.clone(), weight: diam };

        if matching.lacks_a_match_for_row_index( &birth_column ) {
            return Err( PyErr::new::<pyo3::exceptions::PyValueError, _>( "The birth simplex provided has no corresponding death simplex.\nThis message is generated by OAT." ) );
        }

        let death_simplex        =   matching.column_index_for_row_index( &birth_column ).unwrap();
        let death_dimension     =   death_simplex.dimension();
        let death_filtration    =   death_simplex.filtration();
        let b                   =   self.differential_umatch.differential_comb().column_reverse( & death_simplex );

        // incides of a
        let column_indices
            =   self.differential_umatch.boundary_matrix()
                    .cliques_in_lexicographic_order_fixed_dimension( death_dimension as isize )
                    .filter(
                        |x|
                        ( x.filtration() <=   death_filtration )
                        &&
                        matching.lacks_a_match_for_column_index(&x) // exclude positive simplices; in particular, this excluds the death simplex
                    );

        // solve
        let optimized = oat_rust::utilities::optimization::minimize_l1::minimize_l1(
            a, 
            b, 
            obj_fn, 
            column_indices,
            verbose
        ).unwrap();

        // formatting
        let to_ratio = |x: f64| -> Ratio<isize> { Ratio::<isize>::approximate_float(x).unwrap() };
        let format_chain = |x: Vec<_>| {
            let mut r = x
                .into_iter()
                .map(|(k,v): (WeightedSimplex<_>,f64) | (k,to_ratio(v)))
                .collect_vec();
            // r.sort_by( |&(k,v), &(l,u)| order_operator.judge_cmp(&l, &k) );
            r.sort_by( |a,b| order_operator.judge_cmp(a, b) );
            r
        };

        // optimal solution data     
        let chain_optimal               =     format_chain( optimized.y().clone() );
        let mut chain_initial           =     optimized.b().clone();        
        chain_initial.sort();

        let objective_old               =   optimized.cost_b().clone();
        let objective_min               =   optimized.cost_y().clone();


        //  CHECK THE RESULTS
        //  --------------------

        let boundary_initial            =   chain_initial.iter().cloned().multiply_self_as_a_column_vector_with_matrix_and_return_entries_in_reverse_order( boundary_matrix.clone() ).collect_vec();
        let boundary_optimal            =   chain_optimal.iter().cloned().multiply_self_as_a_column_vector_with_matrix_and_return_entries_in_reverse_order( boundary_matrix.clone() ).collect_vec();
        let diff                        =   boundary_initial.iter().cloned().peekable().subtract(
                                                boundary_optimal.iter().cloned().peekable(),
                                                self.differential_umatch.ring_operator(),
                                                boundary_matrix.order_operator_for_row_entries_reverse(),
                                            )
                                            .map( |x| x.1.abs() )
                                            .max();
        println!("max difference in boundaries: {:?}", diff);
        // assert_eq!( boundary_initial, boundary_optimal ); // ensures that the initial and optimal chains have equal boundary


        let dict = PyDict::new(py);

        // row labels
        dict.set_item(
            "type of chain", 
            vec![
                "initial bounding chain", 
                "optimal bounding chain", 
            ]
        )?;

        // objective costs
        dict.set_item(
            "cost", 
            vec![ 
                Some(objective_old), 
                Some(objective_min), 
            ] 
        )?; 

        // number of nonzero entries per vector       
        dict.set_item(
            "number_of_nonzero_coefficients", 
            vec![ 
                chain_initial.len(), 
                chain_optimal.len(), 
            ] 
        )?;

        // vectors
        dict.set_item(
            "chain", 
            vec![ 
                chain_initial.clone().into_dataframe_format(py).ok(), 
                chain_optimal.clone().into_dataframe_format(py).ok(), 
                ] 
        )?;   

        let pandas = py.import("pandas")?;       
        let dict = pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::< Py<PyAny> >::into)?;
        let kwarg = vec![("inplace", true)].into_py_dict(py)?;        
        dict.call_method( py, "set_index", ( "type of chain", ), Some(&kwarg))?;            

        return Ok( dict )
    }        



    /// Minimize a bounding chain, subject to the constraint that the new bounding chain
    /// has equal boundary, and is obtained by adding simplices born strictly before the
    /// last simplex in the input bounding chain in the refined filtration order.
    /// 
    /// The constraint is formalized as follows, where `death_simplex` is the death simplex
    /// that pairs with the user-provided `birth_simplex`. We say that `death_simplex`
    /// precedes simplex `sigma` if `death_simplex` precedes `sigma` in the filtration
    /// order on simplices (with ties broken by lexicographic order).
    /// 
    /// 
    ///     minimize the L1 norm of `b + x`
    ///     
    ///     subject to the condition that 
    ///         Dx = 0, and
    ///         x_sigma = 0 for all i such that `death_simplex` precedes simplex `sigma` as described above.
    /// 
    /// 
    /// We suspect this is less efficient that other formulations.
    #[pyo3(signature = (birth_simplex, problem_type,))]
    pub fn optimize_bounding_chain_kernel< 'py >( 
                &self,
                birth_simplex:                      Vec< u16 >,
                problem_type:                       Option< &str >,
                py: Python< 'py >,
            ) -> PyResult< PyObject > { // MinimalCyclePyWeightedSimplexRational {

        // inputs
        let boundary_matrix                   =   self.differential_umatch.boundary_matrix();        
        let matching                  =   self.differential_umatch.generalized_matching_matrix();        
        let order_operator                  =   boundary_matrix.order_operator_for_row_entries_reverse();
        
        // matrix a, vector c, and the dimension function
        let dim_fn = |x: &WeightedSimplex<FiltrationValue> | x.dimension() as isize;
        let obj_fn = |x: &WeightedSimplex<FiltrationValue> | x.filtration().into_inner(); 
        let a = |k: &WeightedSimplex<FiltrationValue>| boundary_matrix.column( & k ); // columns of A are columns of the boundary matrix
             
        // column b
        let diam = self.differential_umatch.boundary_matrix().filtration_value_for_clique(&birth_simplex).unwrap();
        let birth_column = WeightedSimplex{ vertices: birth_simplex.clone(), weight: diam };

        if matching.lacks_a_match_for_row_index( &birth_column ) {
            return Err( PyErr::new::<pyo3::exceptions::PyValueError, _>( "The birth simplex provided by the user has no corresponding death simplex. This implies that the corresponding cycle is not a boundary; therefore it is infeasible to return a bounding chain.\n\nThis message is generated by OAT." ) );
        }

        let death_column        =   matching.column_index_for_row_index( &birth_column ).unwrap();
        let death_dimension     =   death_column.dimension();
        let death_filtration    =   death_column.filtration();
        let b                   =   self.differential_umatch.differential_comb().column_reverse( & death_column ).collect_vec();




        let column_indices =                            self.differential_umatch.boundary_matrix()
                            .cliques_in_lexicographic_order_fixed_dimension( death_dimension as isize )
                            .filter(
                                |x|
                                ( x.filtration() <=   death_filtration )
                                &&
                                ( x != &death_column )
                            );

        // solve
        let optimized = oat_rust::utilities::optimization::minimize_l1::minimize_l1_kernel(a, b, obj_fn, column_indices).unwrap();

        // formatting
        let to_ratio = |x: f64| -> Ratio<isize> { Ratio::<isize>::approximate_float(x).unwrap() };
        let format_chain = |x: Vec<_>| {
            let mut r = x
                .into_iter()
                .map(|(k,v): (WeightedSimplex<_>,f64) | (k,to_ratio(v)))
                .collect_vec();
            // r.sort_by( |&(k,v), &(l,u)| order_operator.judge_cmp(&l, &k) );
            r.sort_by( |a,b| order_operator.judge_cmp(a, b) );
            r
        };
        
        // optimal solution data
        let x                           =     format_chain( optimized.x().clone() );       
        let chain_optimal               =     format_chain( optimized.y().clone() );
        let mut chain_initial           =     optimized.b().clone();        
        chain_initial.sort();

        let objective_old               =   optimized.cost_b().clone();
        let objective_min               =   optimized.cost_y().clone();


        //  CHECK THE RESULTS
        //  --------------------

        // let boundary_initial            =   chain_initial.iter().cloned().multiply_self_as_a_column_vector_with_matrix_and_return_entries_in_reverse_order( differential.clone() ).collect_vec();
        // let boundary_optimal            =   chain_optimal.iter().cloned().multiply_self_as_a_column_vector_with_matrix_and_return_entries_in_reverse_order( differential.clone() ).collect_vec();
        // assert_eq!( boundary_initial, boundary_optimal ); // ensures that the initial and optimal chains have equal boundary


        let dict = PyDict::new(py);

        // row labels
        dict.set_item(
            "type of chain", 
            vec![
                "initial bounding chain", 
                "optimal bounding chain", 
                "surface_between_cycles", 
            ]
        )?;

        // objective costs
        dict.set_item(
            "cost", 
            vec![ 
                Some(objective_old), 
                Some(objective_min), 
                None, 
            ] 
        )?; 

        // number of nonzero entries per vector       
        dict.set_item(
            "number_of_nonzero_coefficients", 
            vec![ 
                chain_initial.len(), 
                chain_optimal.len(), 
                x.len(), 
            ] 
        )?;

        // vectors
        dict.set_item(
            "chain", 
            vec![ 
                chain_initial.clone().into_dataframe_format(py)?, 
                chain_optimal.clone().into_dataframe_format(py)?, 
                x.clone().into_dataframe_format(py)?, 
                ] 
        )?;   

        let pandas = py.import("pandas")?;       
        let dict = pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::< Py<PyAny> >::into)?;
        let kwarg = vec![("inplace", true)].into_py_dict(py)?;        
        dict.call_method( py, "set_index", ( "type of chain", ), Some(&kwarg));        

        return Ok(dict)
    }        









  

}






// THIS IMPLEMENTATION BLOCK PROVIDES BASIC RUST FUNCTIONS WHICH ARE MIRRORED FOR PYTHON
//
// We keep them separate from their Python counterparts in case we decide to move them
// elsewhere in the future. 
impl DifferentialUmatchVietorisRipsPython {
    





    /// Returns a set of row/column indices of the filtered boundary matrix relevant to homology in dimensions 0 .. d (inclusive), in sorted order
    /// 
    /// If the max homology dimension passed by the user when factoring the boundary matrix is `d`, then
    /// the indices include
    /// - every simplex of dimension `<= d`, and 
    /// - every simplex of dimension `d+1` that indexes a nonzero column of the generalized matching matrix in the :term:`differential Umatch decompostion`
    pub fn boundary_matrix_indices( &self ) 
            ->  Vec<
                         WeightedSimplex<OrderedFloat<f64>> 
                    >
    {
        let mut row_indices  =   self.differential_umatch.row_reduction_indices().clone();     
        let matching    =   self.differential_umatch.generalized_matching_matrix();   

        // we have to reverse the order of row indices (within each dimension) to place things in ascending order
        if row_indices.len() > 0 {
            let mut lhs                     =   0;
            let mut lhs_dim                 =   row_indices[lhs].dimension();
            for  rhs in 0 .. row_indices.len() {
                let rhs_dim                 =   row_indices[rhs].dimension();
                if lhs_dim < rhs_dim { // if we jump a dimension, then reverse the last block of equal-dimension simplices
                    let subset = &mut row_indices[lhs..rhs];
                    subset.reverse(); // reverse the indices of dimension lhs_dim
                    lhs_dim                         =   rhs_dim; // increment the dimension
                    lhs                             =   rhs;
                }
            }
            // the last dimension is an edge case:
            let rhs = row_indices.len();  
            let subset = &mut row_indices[lhs..rhs];
            subset.reverse(); // reverse the indices of dimension lhs_dim
        }    

        // it's possible that some row indices match to some column indices that have dimension > the maximum dimension of
        // any row index.  we don't record max dimension, so we have to calculate it by hand
        if matching.number_of_structural_nonzeros() > 0 {

            // the max dimension of any row index; this works because rows are placed in ascending order of dimension
            let max_row_dimension       =   row_indices.last().unwrap().dimension();

            // the max dimension of a matched column is the dimension of the last matched column index that is stored 
            // internally by the matching matrix, because entries in the matching matrix are stored in ascending order 
            // of row index, and rows are stored in ascending order of dimension 
            let matched_columns         =   matching.bijection_column_indices_to_ordinals_and_inverse().vec_elements_in_order(); // note this is MINOR keys
            let max_col_dimension       =   matched_columns.last().unwrap().dimension(); 

            if max_row_dimension < max_col_dimension { // in this case we need to include some simplices not indexed by our row indices
                // collect just the top-dimensional column indices
                let mut new_simplices       =   matched_columns.iter().filter(|x| x.dimension() == max_col_dimension).cloned().collect_vec();
                new_simplices.sort(); // sort
                row_indices.extend(new_simplices); // add to the row indices
            }
        }

        return row_indices
    }   





 












    /// Returns the Escolar-Hiraoka indices of a persistent cycle.
    /// 
    /// See the documentation for [escolar_hiraoka_indices in `oat_rust`](oat_rust::algebra::matrices::operations::umatch::differential::DifferentialUmatch::escolar_hiraoka_indices] for details.
    pub fn escolar_hiraoka_indices( 
                &self,
                birth_simplex:                      Vec< u16 >,
            ) ->
        Vec<
            WeightedSimplex<OrderedFloat<f64>> 
        >
    {
        
        let dim_fn = |x: &WeightedSimplex<FiltrationValue> | x.dimension() as isize;
        let diam = self.differential_umatch.boundary_matrix().filtration_value_for_clique(&birth_simplex).unwrap();
        let birth_column = WeightedSimplex{ vertices: birth_simplex.clone(), weight: diam };

        let column_indices = 
                self.differential_umatch
                    .escolar_hiraoka_indices( & birth_column, dim_fn, ); // indices of all boundary vectors in the jordan basis

        return column_indices
    }    






    /// Returns the boundary matrix, formatted as a `CsMatBase<Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>>>` matrix
    /// 
    /// 
    /// - Only rows/columns that are in the list `self.boundary_matrix_indices()` are included; see the documentation of that function for details.
    /// - The ith row/column of this matrix corresponds to the ith simplex in `self.boundary_matrix_indices()`.
    pub fn boundary_matrix_csmat_base( &self, ) -> 
            CsMatBase< Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>> >
        {    

        let row_indices     =   self.boundary_matrix_indices();
        let inverse_bijection: HashMap<_,_>   =   row_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();
        let mapping             =   self.differential_umatch.boundary_matrix();
        let shape                           =   (row_indices.len(), row_indices.len());

        let mut indices_row     =   Vec::new();
        let mut indices_col     =   Vec::new();
        let mut vals     =   Vec::new();

        for index_row in row_indices.iter().cloned() {
            for ( index_col, coefficient ) in mapping.row(&index_row) {
                if inverse_bijection.contains_key( &index_col ) { // we screen out columns that are not in our index set
                    indices_row.push( inverse_bijection[&index_row.clone()].clone() );
                    indices_col.push( inverse_bijection[&index_col        ].clone() );                
                    vals.push( coefficient );
                }
            }
        }

        let mat                 =   TriMatBase::from_triplets(shape, indices_row, indices_col, vals);
        let mat                             =   mat.to_csr();
        return mat
    }    








    /// Returns the differential COBM `J` of a [differential Umatch decomposition](oat_rust::algebra::matrices::operations::umatch::differential)
    /// `JM = DJ`.
    /// 
    /// - Here `D` stands for the differential matrix (aka boundary matrix) returned by `self.boundary_matrix_csmat_base()`.
    /// - The ith row/column of `J` corresponds to the ith simplex in `self.boundary_matrix_indices()`.
    /// - `J` is returned as a [CsMatBase] sparse matrix
    pub fn differential_comb_csmat_base( &self, ) -> 
            CsMatBase< Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>> >
        {    

        let row_indices     =   self.boundary_matrix_indices();
        let inverse_bijection: HashMap<_,_>   =   row_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();
        let comb                            =   self.differential_umatch.differential_comb();
        let shape                           =   (row_indices.len(), row_indices.len());

        let mut indices_row     =   Vec::new();
        let mut indices_col     =   Vec::new();
        let mut vals     =   Vec::new();

        for index_row in row_indices.iter().cloned() {
            for ( index_col, coefficient ) in comb.row(&index_row) {
                if inverse_bijection.contains_key( &index_col ) { // we screen out columns that are not in our index set
                    indices_row.push( inverse_bijection[&index_row.clone()].clone() );
                    indices_col.push( inverse_bijection[&index_col        ].clone() );                
                    vals.push( coefficient );
                }

            }
        }

        let mat                 =   TriMatBase::from_triplets(shape, indices_row, indices_col, vals);
        let mat                             =   mat.to_csr();
        return mat

    }      




    /// Returns the *inverse* of the differential COBM `J` of a [differential Umatch decomposition](oat_rust::algebra::matrices::operations::umatch::differential)
    /// `JM = DJ`.
    /// 
    /// - Here `D` stands for the differential matrix (aka boundary matrix) returned by `self.boundary_matrix_csmat_base()`.
    /// - The ith row/column of `J` corresponds to the ith simplex in `self.boundary_matrix_indices()`.
    /// - `Jinv` is returned as a [CsMatBase] sparse matrix
    pub fn differential_comb_inverse_csmat_base( &self, ) -> 
            CsMatBase< Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>> >
        {    

        let row_indices     =   self.boundary_matrix_indices();
        let inverse_bijection: HashMap<_,_>   =   row_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();
        let comb_inverse                            =   self.differential_umatch.differential_comb_inverse();
        let shape                           =   (row_indices.len(), row_indices.len());

        let mut indices_row     =   Vec::new();
        let mut indices_col     =   Vec::new();
        let mut vals     =   Vec::new();

        for index_row in row_indices.iter().cloned() {
            for ( index_col, coefficient ) in comb_inverse.row(&index_row) {
                if inverse_bijection.contains_key( &index_col ) { // we screen out columns that are not in our index set
                    indices_row.push( inverse_bijection[&index_row.clone()].clone() );
                    indices_col.push( inverse_bijection[&index_col        ].clone() );                
                    vals.push( coefficient );
                }

            }
        }

        let mat                 =   TriMatBase::from_triplets(shape, indices_row, indices_col, vals);
        let mat                             =   mat.to_csr();
        return mat

    }      




    /// Returns the generalized matching matrix `M` of a [differential Umatch decomposition](oat_rust::algebra::matrices::operations::umatch::differential)
    /// `JM = DJ`.
    /// 
    /// - Here `D` stands for the differential matrix (aka boundary matrix) returned by `self.boundary_matrix_csmat_base()`.
    /// - The ith row/column of each matrix (`J, M, D`) corresponds to the ith simplex in `self.boundary_matrix_indices()`.
    /// - `M` is returned as a [CsMatBase] sparse matrix
    pub fn generalized_matching_matrix_csmat_base( &self, ) -> 
            CsMatBase< Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>> >
        {    

        // get a list of row indices
        let row_indices     =   self.boundary_matrix_indices();
        // this hashmap maps each simplex to its index in the list of row indices
        let inverse_bijection: HashMap<_,_>   =   row_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();
        let matching                            =   self.differential_umatch.generalized_matching_matrix();
        let shape                           =   (row_indices.len(), row_indices.len());

        let mut indices_row     =   Vec::new();
        let mut indices_col     =   Vec::new();
        let mut vals     =   Vec::new();

        for index_row in row_indices.iter().cloned() {
            for ( index_col, coefficient ) in matching.row(&index_row) {
                if inverse_bijection.contains_key( &index_col ) { // we screen out columns that are not in our index set
                    indices_row.push( inverse_bijection[&index_row.clone()].clone() );
                    indices_col.push( inverse_bijection[&index_col        ].clone() );                
                    vals.push( coefficient );
                }

            }
        }

        let mat                 =   TriMatBase::from_triplets(shape, indices_row, indices_col, vals);
        let mat                             =   mat.to_csr();
        return mat

    }      








}















// #[pyfunction]
// pub fn get_factored_vr_complex( 
//             dissimilarity_matrix: Vec<Vec<f64>>, 
//             max_homology_dimension: isize,
//             max_dissimilarity: Option<f64>,
//         ) 
//     ->  VietorisRipsComplexDifferentialUmatch
//     // DifferentialUmatch<
//     //             OracleDeref< 
//     //                     VietorisRipsComplex<
//     //                             OracleDeref< CsMatBase< FiltrationValue, usize, Vec<usize>, Vec<usize>, Vec<FiltrationValue> > >,
//     //                             FiltrationValue, 
//     //                             RingElement, 
//     //                             RingOperatorForNativeRustNumberType<RingElement>
//     //                         >                
//     //                 >,
//     //             RingOperatorForNativeRustNumberType< RingElement >,
//     //             OrderOperatorByKey< 
//     //                     WeightedSimplex< FiltrationValue >,
//     //                     RingElement,
//     //                     ( WeightedSimplex< FiltrationValue >, RingElement ) 
//     //                 >,
//     //             WeightedSimplex< FiltrationValue >,
//     //             ( WeightedSimplex< FiltrationValue >, RingElement ) ,
//     //             Cloned<Iter<WeightedSimplex<FiltrationValue>>>
//     //         >
//     {

//     println!("TODO: retool barcode to return a dataframe");      

//     let n_points = dissimilarity_matrix.len();  
 
//     // convert the dissimilarity matrix to type FiltrationValue
//     let dissimilarity_matrix_data
//         =   dissimilarity_matrix.iter().map(|x| x.iter().cloned().map(|x| OrderedFloat(x)).collect_vec() )
//             .collect_vec().into_csr( n_points, n_points );
//     let dissimilarity_matrix = Arc::new( dissimilarity_matrix_data );           
//     let max_dissimilarity = max_dissimilarity.map(|x| OrderedFloat(x));
//     let dissimilarity_min = 
//             { 
//                 if n_points==0 {
//                     OrderedFloat(0.0)
//                 } else { 
//                     dissimilarity_matrix.data().iter().min().unwrap().clone()
//                 } 
//             };               
//     // define the ring operator
//     let ring_operator = FieldRationalSize::new();
//     // define the chain complex
//     let chain_complex_data = VietorisRipsComplex::new( dissimilarity_matrix, n_points, max_dissimilarity, dissimilarity_min, ring_operator.clone() );
//     // get a reference to the chain complex (needed in order to create certain iterators, due to lifetime bounds)
//     // let chain_complex_ref = & chain_complex;   
//     let chain_complex = ChainComplexVrFilteredArc::new( Arc::new( chain_complex_data ) );
//     // define an interator to run over the row indices of the boundary matrix 
//     let row_indices = chain_complex.vr().cliques_in_row_reduction_order( max_homology_dimension );
//     let iter_keymaj = row_indices.iter().cloned();    
//     // obtain a u-match factorization of the boundary matrix
//     let factored = DifferentialUmatch::new(
//             chain_complex, 
//             ring_operator.clone(),             
//             OrderOperatorAutoLt::new(), 
//             row_indices,             
//         );      
//     return VietorisRipsComplexDifferentialUmatch{ factored } // DifferentialUmatch { umatch, row_indices }

// }




















//  =========================================
//  COMPUTE PERSISTENT HOMOLOGY
//  =========================================


//  DEPRECATED
// /// Compute basis of cycle representatives for persistent homology of a VR filtration, over the rationals
// /// 
// /// Computes the persistent homology of the filtered clique complex (ie VR complex)
// /// with dissimilarity matrix `dissimilarity_matrix`, over the field of rational numbers.  
// /// 
// /// - Edges of weight `>= max_dissimilarity` are excluded.
// /// - Homology is computed in dimensions 0 through `max_homology_dimension`, inclusive
// /// 
// /// Returns: `BarcodePyWeightedSimplexRational`
// #[pyfunction]
// pub fn persistent_homology_vr( 
//             dissimilarity_matrix: Vec<Vec<f64>>, 
//             max_homology_dimension: isize,
//             max_dissimilarity: Option<f64>,  
//         ) 
//     -> BarcodePyWeightedSimplexRational
//     {

//     println!("TODO: shift to umatch and retool barcode to return a dataframe");
 
//     // // convert the dissimilarity matrix to type FiltrationValue
//     // let dissimilarity_matrix = dissimilarity_matrix.iter().map(|x| x.iter().cloned().map(|x| OrderedFloat(x)).collect() ).collect();
//     // let max_dissimilarity = max_dissimilarity.map(|x| OrderedFloat(x));
//     // // define the ring operator
//     // let ring_operator = FieldRationalSize::new();
//     // // define the chain complex
//     // let chain_complex = VietorisRipsComplex::new( dissimilarity_matrix, max_dissimilarity, ring_operator.clone() );
//     // // get a reference to the chain complex (needed in order to create certain iterators, due to lifetime bounds)
//     // let chain_complex_ref = & chain_complex;   
//     // // define an interator to run over the row indices of the boundary matrix 
//     // let row_indices = chain_complex.cliques_in_row_reduction_order( max_homology_dimension );
//     // let iter_keymaj = row_indices.iter().cloned();    
//     // // obtain a u-match factorization of the boundary matrix
//     // let umatch = new_umatchrowmajor_with_clearing(
//     //         chain_complex_ref, 
//     //         iter_keymaj.clone(), 
//     //         ring_operator.clone(), 
//     //         OrderOperatorAutoLt::new(), 
//     //         OrderOperatorAutoLt::new(), 
//     //     ); 

//     let dissimilarity_matrix_data = dissimilarity_matrix.iter().map(|x| x.iter().cloned().map(|x| OrderedFloat(x)).collect_vec() ).collect_vec();
//     let dissimilarity_matrix_size = dissimilarity_matrix.len();
//     let max_dissimilarity = max_dissimilarity.map(|x| OrderedFloat(x));
//     let dissimilarity_min = 
//             { 
//                 if dissimilarity_matrix_size==0 {
//                     OrderedFloat(0.0)
//                 } else { 
//                     dissimilarity_matrix_data.iter().map(|x| x.iter()).flatten().min().unwrap().clone()
//                 } 
//             };
//     let dissimilarity_matrix = Arc::new( dissimilarity_matrix_data.into_csr( dissimilarity_matrix_size, dissimilarity_matrix_size ) );
//     // define the ring operator
//     let ring_operator = FieldRationalSize::new();
//     // define the chain complex
//     let chain_complex = VietorisRipsComplex::new( dissimilarity_matrix, dissimilarity_matrix_size, max_dissimilarity, dissimilarity_min, ring_operator.clone() );
//     let factored = chain_complex.factor_from_arc( max_homology_dimension );    

//     // unpack the factored boundary matrix into a barcode
//     let dim_fn = |x: &WeightedSimplex<FiltrationValue> | x.dimension() as isize;
//     let fil_fn = |x: &WeightedSimplex<FiltrationValue> | x.filtration();    
//     let barcode = oat_rust::algebra::chain_complexes::barcode::barcode( factored.umatch(), factored.factored.cliques_in_row_reduction_order(max_homology_dimension), dim_fn, fil_fn, true , true);
      
//     return BarcodePyWeightedSimplexRational::new( barcode )
// }




//  DEPRECATED IN FAVOR OF METHODS ON A FACTORED COMPLEX
// /// persistent_homology_vr_optimized(dissimilarity_matrix: Vec<Vec<f64>>, max_homology_dimension: isize, max_dissimilarity: Option<f64>, /)
// /// --
// ///
// /// Compute basis of *optimized* cycle representatives for persistent homology of a VR filtration, over the rationals
// /// 
// /// Computes the persistent homology of the filtered clique complex (ie VR complex)
// /// with dissimilarity matrix `dissimilarity_matrix`, over the field of rational numbers.  
// /// 
// /// - Edges of weight `>= max_dissimilarity` are excluded.
// /// - Homology is computed in dimensions 0 through `max_homology_dimension`, inclusive
// /// 
// /// Returns: `( BarcodePyWeightedSimplexRational, L )`, where `L[p]` is the optimized cycle for the bar with unique id number `p`.
// #[pyfunction]
// // #[args( dissimilarity_matrix = "vec![vec![]]", max_homology_dimension="0", max_dissimilarity="None", )]
// // #[text_signature = "(dissimilarity_matrix, max_homology_dimension, max_dissimilarity, /)"]
// // #[pyo3(signature = (dissimilarity_matrix=vec![vec![0.0]], max_homology_dimension=0, max_dissimilarity=None))]
// // #[pyo3(signature = (dissimilarity_matrix, max_homology_dimension, max_dissimilarity))]
// pub fn persistent_homology_vr_optimized( 
//             dissimilarity_matrix: Vec<Vec<f64>>, 
//             max_homology_dimension: isize,
//             max_dissimilarity: Option<f64>,
//             dissimilarity_min: Option<f64>,
//         ) 
//     -> ( BarcodePyWeightedSimplexRational, Vec< MinimalCyclePyWeightedSimplexRational > )
//     {

//     println!("###############");
 
//     // convert the dissimilarity matrix to type FiltrationValue
//     let dissimilarity_matrix_data = dissimilarity_matrix.iter().map(|x| x.iter().cloned().map(|x| OrderedFloat(x)).collect_vec() ).collect_vec();
//     let dissimilarity_matrix_size = dissimilarity_matrix.len();
//     let max_dissimilarity = max_dissimilarity.map(|x| OrderedFloat(x));
//     let dissimilarity_min = dissimilarity_min.map_or( 
//             { 
//                 if dissimilarity_matrix_size==0 {
//                     OrderedFloat(0.0)
//                 } else { 
//                     dissimilarity_matrix_data.iter().map(|x| x.iter()).flatten().min().unwrap().clone()
//                 } 
//             },
//             |x: f64| OrderedFloat(x),            
//         );
//     let dissimilarity_matrix = Arc::new( dissimilarity_matrix_data.into_csr( dissimilarity_matrix_size, dissimilarity_matrix_size ) );
//     // define the ring operator
//     let ring_operator = FieldRationalSize::new();
//     // define the chain complex
//     let chain_complex = VietorisRipsComplex::new( dissimilarity_matrix, dissimilarity_matrix_size, max_dissimilarity, dissimilarity_min, ring_operator.clone() );
//     let factored = chain_complex.factor_from_arc( max_homology_dimension );

//     // get a reference to the chain complex (needed in order to create certain iterators, due to lifetime bounds)
//     let chain_complex_ref = & chain_complex;   
//     // define an interator to run over the row indices of the boundary matrix 
//     let row_indices = chain_complex.cliques_in_row_reduction_order( max_homology_dimension );
//     let iter_keymaj = row_indices.iter().cloned();    
//     // obtain a u-match factorization of the boundary matrix
//     let factored = DifferentialUmatch::new(
//                 boundary_matrix, 
//                 ring_operator, 
//                 order_comparator, 
//                 row_indices
//             );

//     let umatch = new_umatchrowmajor_with_clearing(
//             chain_complex_ref, 
//             iter_keymaj.clone(), 
//             ring_operator.clone(), 
//             OrderOperatorAutoLt::new(), 
//             OrderOperatorAutoLt::new(), 
//         );      
//     // unpack the factored boundary matrix into a barcode
//     let dim_fn = |x: &WeightedSimplex<FiltrationValue> | x.dimension() as isize;
//     let fil_fn = |x: &WeightedSimplex<FiltrationValue> | x.filtration();    
//     let obj_fn = |x: &WeightedSimplex<FiltrationValue> | x.filtration().into_inner();                
//     let barcode = oat_rust::algebra::chain_complexes::barcode::barcode( &umatch, iter_keymaj.clone(), dim_fn, fil_fn, true , true);
      
//     let mut optimized_cycles                =   Vec::new();
//     use indicatif::ProgressBar;
//     // let progress_bar = ProgressBar::new( barcode.bars().len() );

//     for bar in barcode.iter() {

//         // progress_bar.inc(1);

//         if bar.dimension() == 0 { continue }
        
//         let birth_column                =   bar.birth_column();
//         let optimized                   =   minimize_cycle(
//                                                     & umatch,
//                                                     iter_keymaj.clone(),
//                                                     dim_fn,
//                                                     fil_fn,
//                                                     obj_fn,
//                                                     birth_column.clone(),
//                                                 )?;   
//         let cycle_old                   =   optimized.cycle_initial().iter().cloned()
//                                                 .map(|(simplex,coeff)| ( WeightedSimplexPython::new(simplex), coeff ) ).collect();
//         let cycle_min                   =   optimized.cycle_optimal().iter().cloned()
//                                                 .map(|(simplex,coeff)| ( WeightedSimplexPython::new(simplex), coeff ) ).collect();
//         let bounding_chain              =   optimized.bounding_chain().iter().cloned()
//                                                 .map(|(simplex,coeff)| ( WeightedSimplexPython::new(simplex), coeff ) ).collect();
//         let objective_old               =   optimized.objective_initial().clone();
//         let objective_min               =   optimized.objective_optimal().clone();

//         let optimized                   =   MinimalCycle::new(cycle_old, cycle_min, objective_old, objective_min, bounding_chain);
//         let optimized                   = MinimalCyclePyWeightedSimplexRational::new( optimized );
//         optimized_cycles.push( optimized );
//     }
//     let barcode = BarcodePyWeightedSimplexRational::new( barcode );

//     return (barcode, optimized_cycles)
// }


#[cfg(test)]
mod tests {
    use pyo3::Python;
    use oat_rust::algebra::matrices::types::third_party::IntoCSR;

    


    // #[test]
    // fn test_barcode_fixed_symmetric_matrix() {

    //     use crate::clique_filtered::VietorisRipsComplexDifferentialUmatch;

    //     let max_dissimilarity = None;
    //     let max_homology_dimension = Some(1);

    //     let dissimilarity_matrix =
    //     vec![ 
    //     vec![0.0, 0.6016928528850207, 0.493811064571812, 0.7631842110599732, 0.6190969952854828, 0.32238439536052743, 0.5577776299243353, 0.7818974214708962, 0.07661198884101905, 0.4725681975471917, 0.11373899464129633, 0.42692474128277236, 0.8617605210898125, 0.6033834157784794, 0.6507666017239748, 0.6108287386340484, 0.6874754930701601, 0.5216650170561481, 0.1739545434174833, 0.3848087421417594],
    //     vec![0.6016928528850207, 0.0, 0.5092128196637472, 0.3972421208618373, 0.3046297569686842, 0.4124608436158862, 0.2806048596469476, 0.3519192500394136, 0.5956941890831011, 0.3891213477906711, 0.05217685800395466, 0.5673170383785954, 0.6154346905039156, 0.8410186822326671, 0.6106959601576187, 0.7283439354447504, 0.5496200412544044, 0.5000451211467285, 0.3798535242449169, 0.5243930541547187],
    //     vec![0.493811064571812, 0.5092128196637472, 0.0, 0.3561773339990194, 0.34386022814969286, 0.47820995353849394, 0.3358482108698321, 0.3112545444910565, 0.6769811259281259, 0.11951440156345605, 0.28557067972725503, 0.512837799856345, 0.14341566187913501, 0.19856500421639478, 0.5350631971916313, 0.7224583474471165, 0.6061450244826808, 0.9072555593504178, 0.42069193806319394, 0.6319175411184014],
    //     vec![0.7631842110599732, 0.3972421208618373, 0.3561773339990194, 0.0, 0.3487932780695402, 0.46719510926568875, 0.5104490109819306, 0.42786488797344424, 0.7260539344838907, 0.5216838489415861, 0.3665979132978837, 0.1739258675892733, 0.4606050949827942, 0.40558969160305447, 0.5658659589949734, 0.22907682914861116, 0.8173301082204779, 0.320916283257647, 0.42774123820610455, 0.8634899734150683],
    //     vec![0.6190969952854828, 0.3046297569686842, 0.34386022814969286, 0.3487932780695402, 0.0, 0.7662218806020976, 0.46151054210129994, 0.4990724633689937, 0.8707390402111069, 0.3505745678194895, 0.5189917539728075, 0.42824076055710325, 0.1842675961210688, 0.8458600272472917, 0.24228395929341928, 0.14468843668941522, 0.28900265054271523, 0.5305753485287981, 0.5462378624754798, 0.8095581217994358],
    //     vec![0.32238439536052743, 0.4124608436158862, 0.47820995353849394, 0.46719510926568875, 0.7662218806020976, 0.0, 0.3845952142742255, 0.2536554291638623, 0.6413595103144578, 0.6128904410415779, 0.5285348472099765, 0.5177751670806997, 0.711864519210214, 0.62428815517063, 0.19414566417205048, 0.7066233019294025, 0.43615946930926086, 0.6186740371220466, 0.18308366800056086, 0.6834811848495779],
    //     vec![0.5577776299243353, 0.2806048596469476, 0.3358482108698321, 0.5104490109819306, 0.46151054210129994, 0.3845952142742255, 0.0, 0.694751663136327, 0.3474705875777475, 0.26206817949657735, 0.6336863206261203, 0.26798771265418375, 0.14444456010669526, 0.6854355294928525, 0.09457649870433515, 0.7190028894889605, 0.037081784782752036, 0.37413897799597495, 0.4989135518265708, 0.3728811748113052],
    //     vec![0.7818974214708962, 0.3519192500394136, 0.3112545444910565, 0.42786488797344424, 0.4990724633689937, 0.2536554291638623, 0.694751663136327, 0.0, 0.17392906677117737, 0.6210156343133215, 0.5375749239944999, 0.5187858806627833, 0.5929340790641354, 0.7712449339329094, 0.3059336215936842, 0.36033157987432385, 0.28570096380399235, 0.04339918302952661, 0.29419322463799524, 0.2429942786113325],
    //     vec![0.07661198884101905, 0.5956941890831011, 0.6769811259281259, 0.7260539344838907, 0.8707390402111069, 0.6413595103144578, 0.3474705875777475, 0.17392906677117737, 0.0, 0.38767381994292616, 0.458781018569824, 0.4517193143860384, 0.4113984645352643, 0.21272714858166386, 0.4293977593552041, 0.6653615561279136, 0.964931953987687, 0.18254377535411093, 0.28709617555076605, 0.554288129648074],
    //     vec![0.4725681975471917, 0.3891213477906711, 0.11951440156345605, 0.5216838489415861, 0.3505745678194895, 0.6128904410415779, 0.26206817949657735, 0.6210156343133215, 0.38767381994292616, 0.0, 0.5159333989612956, 0.4175055055353978, 0.1623817553586221, 0.2509588162503712, 0.5131209051562422, 0.6430031786739608, 0.7268562340691295, 0.19940288391942473, 0.4270267130780456, 0.5342481723480923],
    //     vec![0.11373899464129633, 0.05217685800395466, 0.28557067972725503, 0.3665979132978837, 0.5189917539728075, 0.5285348472099765, 0.6336863206261203, 0.5375749239944999, 0.458781018569824, 0.5159333989612956, 0.0, 0.12132671005368023, 0.7324788222379005, 0.406730119748273, 0.45044677792578536, 0.9318540754195065, 0.4075271777861631, 0.7319995137475207, 0.15237965124911634, 0.5429616218744323],
    //     vec![0.42692474128277236, 0.5673170383785954, 0.512837799856345, 0.1739258675892733, 0.42824076055710325, 0.5177751670806997, 0.26798771265418375, 0.5187858806627833, 0.4517193143860384, 0.4175055055353978, 0.12132671005368023, 0.0, 0.49532472161425556, 0.3020632653745947, 0.6646579793145441, 0.2693091880632087, 0.386742413025264, 0.2831998326688344, 0.3599502190526389, 0.4935662425765617],
    //     vec![0.8617605210898125, 0.6154346905039156, 0.14341566187913501, 0.4606050949827942, 0.1842675961210688, 0.711864519210214, 0.14444456010669526, 0.5929340790641354, 0.4113984645352643, 0.1623817553586221, 0.7324788222379005, 0.49532472161425556, 0.0, 0.49055724427179814, 0.7323387041095746, 0.25285282889479155, 0.5228054905023033, 0.5501041781782425, 0.4691772921034907, 0.4847299731552148],
    //     vec![0.6033834157784794, 0.8410186822326671, 0.19856500421639478, 0.40558969160305447, 0.8458600272472917, 0.62428815517063, 0.6854355294928525, 0.7712449339329094, 0.21272714858166386, 0.2509588162503712, 0.406730119748273, 0.3020632653745947, 0.49055724427179814, 0.0, 0.046310411737922164, 0.48601695582724214, 0.3806904221812635, 0.6554292411367946, 0.3304760871094675, 0.4383023912725962],
    //     vec![0.6507666017239748, 0.6106959601576187, 0.5350631971916313, 0.5658659589949734, 0.24228395929341928, 0.19414566417205048, 0.09457649870433515, 0.3059336215936842, 0.4293977593552041, 0.5131209051562422, 0.45044677792578536, 0.6646579793145441, 0.7323387041095746, 0.046310411737922164, 0.0, 0.15570429291859234, 0.6035507808993115, 0.627016949499856, 0.42846636792455217, 0.8690711833626937],
    //     vec![0.6108287386340484, 0.7283439354447504, 0.7224583474471165, 0.22907682914861116, 0.14468843668941522, 0.7066233019294025, 0.7190028894889605, 0.36033157987432385, 0.6653615561279136, 0.6430031786739608, 0.9318540754195065, 0.2693091880632087, 0.25285282889479155, 0.48601695582724214, 0.15570429291859234, 0.0, 0.4771858779979862, 0.44438375123613827, 0.34983216058393884, 0.8142058135029405],
    //     vec![0.6874754930701601, 0.5496200412544044, 0.6061450244826808, 0.8173301082204779, 0.28900265054271523, 0.43615946930926086, 0.037081784782752036, 0.28570096380399235, 0.964931953987687, 0.7268562340691295, 0.4075271777861631, 0.386742413025264, 0.5228054905023033, 0.3806904221812635, 0.6035507808993115, 0.4771858779979862, 0.0, 0.48393447385845423, 0.6526221039553707, 0.17013104544474267],
    //     vec![0.5216650170561481, 0.5000451211467285, 0.9072555593504178, 0.320916283257647, 0.5305753485287981, 0.6186740371220466, 0.37413897799597495, 0.04339918302952661, 0.18254377535411093, 0.19940288391942473, 0.7319995137475207, 0.2831998326688344, 0.5501041781782425, 0.6554292411367946, 0.627016949499856, 0.44438375123613827, 0.48393447385845423, 0.0, 0.051247063916904145, 0.5188480070944168],
    //     vec![0.1739545434174833, 0.3798535242449169, 0.42069193806319394, 0.42774123820610455, 0.5462378624754798, 0.18308366800056086, 0.4989135518265708, 0.29419322463799524, 0.28709617555076605, 0.4270267130780456, 0.15237965124911634, 0.3599502190526389, 0.4691772921034907, 0.3304760871094675, 0.42846636792455217, 0.34983216058393884, 0.6526221039553707, 0.051247063916904145, 0.0, 0.707666916030988],
    //     vec![0.3848087421417594, 0.5243930541547187, 0.6319175411184014, 0.8634899734150683, 0.8095581217994358, 0.6834811848495779, 0.3728811748113052, 0.2429942786113325, 0.554288129648074, 0.5342481723480923, 0.5429616218744323, 0.4935662425765617, 0.4847299731552148, 0.4383023912725962, 0.8690711833626937, 0.8142058135029405, 0.17013104544474267, 0.5188480070944168, 0.707666916030988, 0.0],
    //     ];
    //     let dissimilarity_matrix = dissimilarity_matrix.into_csr(dissimilarity_matrix.len(),dissimilarity_matrix.len(),);

    //     Python::with_gil(|py| {
    //         let factored = VietorisRipsComplexDifferentialUmatch::new(py, dissimilarity_matrix, max_dissimilarity,max_homology_dimension);   
    //     });

        
    
    // }
 
}