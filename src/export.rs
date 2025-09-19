//! Export data to Python
//! 
//! Provides wrappers for exporting data to Python.
//! 
//! There are wrappers to export to dataframes, matrices, dictionaries, etc. This is handy, since you can choose
//! the format you want to export, then use the corresponding wrapper.
//! 

use std::sync::Arc;

use itertools::Itertools;
use num::rational::Ratio;
use num::ToPrimitive;
use oat_rust::algebra::rings::types::native::RingOperatorForNativeRustNumberType;
use oat_rust::topology::simplicial::from::graph_weighted::VietorisRipsComplex;
use ordered_float::OrderedFloat;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use oat_rust::topology::simplicial::simplices::weighted::WeightedSimplex;
use sprs::CsMatBase;












//  =========================================================
//  THE INTO TUPLES TRAIT
//  =========================================================




pub trait IntoVecOfPyTuples where Self: Sized {
    fn into_vec_of_py_tuples(self, py: Python<'_>) -> PyResult< Vec<pyo3::Bound<'_, PyTuple, >> >;
}


impl IntoVecOfPyTuples for 
    Vec< Vec< u16 > >
{
    /// Converts `self` into a vector of tuples
    fn into_vec_of_py_tuples(self, py: Python<'_>) -> PyResult< Vec<pyo3::Bound<'_, PyTuple, >> > {
        let mut tuples = Vec::with_capacity(self.len());
        for simplex in self.iter() {
            let tuple: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            tuples.push(tuple);
        }
        Ok(tuples)
    }
}





//  =========================================================
//  THE SCIPY CSR TRAIT
//  =========================================================

/// Provides a method `into_scipy_csr_format` for any type `T`
pub trait IntoScipyCsrFormat where Self: Sized {
    fn into_scipy_csr_format(self, py: Python<'_>) -> PyResult<PyObject>;   // ScipyCsrExportWrapper< Self >;   
}

// /// Wrapper for matrices we wish to export into Scipy CSR format
// pub struct ScipyCsrExportWrapper< T > {
//     pub data: T,
// }








//  =========================================================
//  THE EXPORT TRAIT
//  =========================================================


/// Generic wrapper used to export objects to Python
/// 
/// It's often necessary to wrap an object `t: T` in a wrapper struct
/// `Wrapper` in order to implement the PyO3 `IntoPy` or `IntoPyObject` traits.
/// This struct offers a convenient way to do so: for given `T`:
/// 1. Implement `IntoPy` or `IntoPyObject` on `ForExport<T>`
/// 2. Then convert any object `t: T` to an object `wrapper: ForExport< T >`
/// via the `Export` trait; concretely `wrapper = t.export()`.
#[derive(Copy,Clone,Debug,Eq,PartialEq,Ord,PartialOrd)]
#[derive(IntoPyObject)]
pub struct ForExport< T > {
    pub data: T,
}

/// Provides a method `export` for any type `T`
pub trait Export where Self: Sized {
    fn export( self ) -> ForExport< Self >;   
}

impl < T > Export for T {
    fn export( self ) -> ForExport< Self > { ForExport{ data: self } }    
}








//  =========================================================
//  THE DATAFRAME TRAIT
//  =========================================================

/// Provides a method `into_dataframe` for any type `T`
pub trait IntoDataframeFormat where Self: Sized {
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject>;   
}










//  =========================================================
//  Vec< (WeightedSimplex, Bound) >
//  =========================================================


impl < 'py >IntoDataframeFormat for 
        
    Vec< ( WeightedSimplex< OrderedFloat<f64> >, Bound<'py, PyAny> ) >
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut simplices = Vec::with_capacity(self.len());
        for entry in self.iter() {
            let simplex = entry.0.vertices();
            let simplex: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            simplices.push(simplex);
        }
        dict.set_item( "simplex",     simplices)?;
        dict.set_item( "filtration",  self.iter().map( |(s,_)| s.filtration().into_inner() ).collect_vec() )?;
        dict.set_item( "coefficient", self.iter().map( |(_,z)| z.clone() ).collect_vec() )?;
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None).map(Into::into)
    }
}





//  =========================================================
//  Vec< (WeightedSimplex, bool) >
//  =========================================================


impl < 'py >IntoDataframeFormat for 
        
    Vec< ( WeightedSimplex< OrderedFloat<f64> >, bool ) >
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut simplices = Vec::with_capacity(self.len());
        for entry in self.iter() {
            let simplex = entry.0.vertices();
            let simplex: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            simplices.push(simplex);
        }
        dict.set_item( "simplex",     simplices)?;
        dict.set_item( "filtration",  self.iter().map( |(s,_)| s.filtration().into_inner() ).collect_vec() )?;
        dict.set_item( "coefficient", self.iter().map( |(_,z)| z.clone() ).collect_vec() )?;
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None).map(Into::into)
    }
}




//  =========================================================
//  Vec< (WeightedSimplex, bool) >
//  =========================================================


impl < 'py >IntoDataframeFormat for 
        
    Vec< ( WeightedSimplex< OrderedFloat<f64> >, isize ) >
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut simplices = Vec::with_capacity(self.len());
        for entry in self.iter() {
            let simplex = entry.0.vertices();
            let simplex: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            simplices.push(simplex);
        }
        dict.set_item( "simplex",     simplices)?;
        dict.set_item( "filtration",  self.iter().map( |(s,_)| s.filtration().into_inner() ).collect_vec() )?;
        dict.set_item( "coefficient", self.iter().map( |(_,z)| z.clone() ).collect_vec() )?;
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None).map(Into::into)
    }
}





//  =========================================================
//  Vec< (WeightedSimplex, Ratio) >
//  =========================================================


impl IntoDataframeFormat for 
        
    Vec< ( WeightedSimplex< OrderedFloat<f64> >, Ratio< isize > ) >
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut simplices = Vec::with_capacity(self.len());
        for entry in self.iter() {
            let simplex = entry.0.vertices();
            let simplex: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            simplices.push(simplex);
        }
        dict.set_item( "simplex",     simplices)?;
        dict.set_item( "filtration",  self.iter().map( |(s,_)| s.filtration().into_inner() ).collect_vec() )?;
        dict.set_item( "coefficient", self.iter().map( |(_,z)| z.clone() ).collect_vec() )?;
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None).map(Into::into)
    }
}






//  =========================================================
//  Vec< WeightedSimplex >
//  =========================================================


impl IntoDataframeFormat for   

        Vec< 
            WeightedSimplex< OrderedFloat<f64> >
        >  
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut simplices = Vec::with_capacity(self.len());
        for simplex in self.iter() {
            let simplex = simplex.vertices();
            let simplex: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            simplices.push(simplex);
        }
        dict.set_item( "simplex",     simplices ).ok().unwrap();
        dict.set_item( "filtration",  self.iter().map( |s| s.filtration().into_inner() ).collect_vec() ).ok().unwrap();
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None).map(Into::into)
    }
}


//  =========================================================
//  Vec< (WeightedSimplex, f64) >
//  =========================================================


impl IntoDataframeFormat for   
        
    Vec< ( WeightedSimplex< OrderedFloat<f64> >, f64 ) > 
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut simplices = Vec::with_capacity(self.len());
        for (simplex,_coefficient) in self.iter() {
            let simplex = simplex.vertices();
            let simplex: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            simplices.push(simplex);
        }
        dict.set_item( "simplex",     simplices ).ok().unwrap();
        dict.set_item( "filtration",  self.iter().map( |(s,_)| s.filtration().into_inner() ).collect_vec() ).ok().unwrap();
        dict.set_item( "coefficient", self.iter().map( |(_,z)| z.clone() ).collect_vec() ).ok().unwrap();
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::into)
    }
}


//  =========================================================
//  FloatChain
//  =========================================================


// impl IntoDataframeFormat for   
        
//     FloatChain
// {
//     /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
//     fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
//         let dict = PyDict::new(py);
//         dict.set_item( "simplex",     self.index_tool.vec_elements_in_order().iter().map( |s| s.vertices() ).collect_vec() ).ok().unwrap();
//         dict.set_item( "filtration",  self.index_tool.vec_elements_in_order().iter().map( |s| s.filtration().into_inner() ).collect_vec() ).ok().unwrap();
//         dict.set_item( "coefficient", self.coefficients.clone()).ok().unwrap();
//         let pandas = py.import("pandas").ok().unwrap();       
//         pandas.call_method("DataFrame", ( dict, ), None)
//             .map(Into::into)
//     }
// }


//  =========================================================
//  Vec< (Vec<isize>, Ratio) >
//  =========================================================


impl IntoDataframeFormat for   
        
    Vec< ( Vec<isize>, Ratio< isize > ) >
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut simplices = Vec::with_capacity(self.len());
        for (simplex,_coefficient) in self.iter() {
            let simplex: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            simplices.push(simplex);
        }
        dict.set_item( "simplex",     simplices ).ok().unwrap();
        dict.set_item( "coefficient", self.iter().map( |(_,z)| z.clone() ).collect_vec() ).ok().unwrap();
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::into)
    }
}

impl IntoDataframeFormat for   
        
    Vec< ( Vec<usize>, Ratio< isize > ) >
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut simplices = Vec::with_capacity(self.len());
        for (simplex,_coefficient) in self.iter() {
            let simplex: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            simplices.push(simplex);
        }        
        dict.set_item( "simplex",     simplices ).ok().unwrap();
        dict.set_item( "coefficient", self.iter().map( |(_,z)| z.clone() ).collect_vec() ).ok().unwrap();
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::into)
    }
}



//  =========================================================
//  Vec< (Vec<isize>, f64) >
//  =========================================================


impl IntoDataframeFormat for   
        
    Vec< ( Vec<isize>, f64 ) >
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut simplices = Vec::with_capacity(self.len());
        for (simplex,_coefficient) in self.iter() {
            let simplex: pyo3::Bound<'_, PyTuple> = PyTuple::new(py, simplex)?;
            simplices.push(simplex);
        }        
        dict.set_item( "simplex",     simplices ).ok().unwrap();
        dict.set_item( "coefficient", self.iter().map( |(_,z)| z.clone() ).collect_vec() ).ok().unwrap();
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::into)
    }
}


impl IntoDataframeFormat for   
        
    Vec< ( Vec<usize>, f64 ) >
{
    /// Returns a Pandas data frame with columns `simplex`, `filtration`, and `coefficient`
    fn into_dataframe_format(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item( "simplex",     self.iter().map( |(s,_)| s.clone() ).collect_vec() ).ok().unwrap();
        dict.set_item( "coefficient", self.iter().map( |(_,z)| z.clone() ).collect_vec() ).ok().unwrap();
        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::into)
    }
}






//  -----------------------------------------------------------------
//  IMPLEMENT EXPORT FOR CSMAT< Ratio<isize> >
//  -----------------------------------------------------------------


impl IntoScipyCsrFormat for 
        
    CsMatBase< 
        Ratio<isize>, 
        usize, 
        Vec<usize>, 
        Vec<usize>, 
        Vec<Ratio<isize>> 
    >
{
    /// Converts `self` into a Scipy CSR matrix
    fn into_scipy_csr_format( self, py: Python<'_>) -> PyResult<PyObject> {

        let shape = self.shape();

        let (indptr, indices, data) = self.into_raw_storage();
        
        let data: Vec<f64> = data.into_iter().map(|r| r.to_f64().unwrap()).collect();


        // We once tried to use the following commented methods to get references to the data without consuming `self`, but ran into some lifetime issues we didn't understand
        // let indices = self.indices();
        // let indptr = self.indptr().raw_storage();
        // let data = self.data();

        let sparse = py.import("scipy.sparse").ok().unwrap();
        return sparse.call_method("csr_matrix", 
            (
                (
                    data,
                    indices,
                    indptr,
                ),
                shape
            ), 
            None
        ).map(Into::into)
    }
}


//  -----------------------------------------------------------------
//  IMPLEMENT EXPORT FOR CSMAT< OrderedFloat<f64> >
//  -----------------------------------------------------------------

impl IntoScipyCsrFormat for 
        
    CsMatBase< 
        OrderedFloat<f64>, 
        usize, 
        Vec<usize>, 
        Vec<usize>, 
        Vec<OrderedFloat<f64>> 
    >
{
    /// Converts `self` into a Scipy CSR matrix
    fn into_scipy_csr_format( self, py: Python<'_>) -> PyResult<PyObject> {

        let shape = self.shape();

        let (indptr, indices, data) = self.into_raw_storage();
        
        let data: Vec<f64> = data.into_iter().map(|r| r.to_f64().unwrap()).collect();


        // We once tried to use the following commented methods to get references to the data without consuming `self`, but ran into some lifetime issues we didn't understand
        // let indices = self.indices();
        // let indptr = self.indptr().raw_storage();
        // let data = self.data();

        let sparse = py.import("scipy.sparse").ok().unwrap();
        return sparse.call_method("csr_matrix", 
            (
                (
                    data,
                    indices,
                    indptr,
                ),
                shape
            ), 
            None
        ).map(Into::into)
    }
}