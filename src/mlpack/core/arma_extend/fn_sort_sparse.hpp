//! \addtogroup fn_sort_sparse
//! @{

/**
 * @file fn_sort_sparse.hpp
 * @author Ivan Georgiev (ivan@jonan.info)
 *
 * Sorting of sparse matrices as extension to arma library.
 */

template <typename ElemType>
SpMat<ElemType> sort(const SpMat<ElemType>& data)
{
  // Construct the vector of values.
  std::vector<ElemType> valsVec(data.begin(), data.end());
  
  // ... and sort it!
  std::sort(valsVec.begin(), valsVec.end());
  
  // Now prepare the structures for the batch construction of the
  // sorted sparse matrix.
  arma::umat locations(2, data.n_nonzero);
  arma::Col<ElemType> vals(data.n_nonzero);
  ElemType lastVal = -std::numeric_limits<ElemType>::max();
  size_t padding = 0;
  
  for (size_t ii = 0; ii < valsVec.size(); ++ii)
  {
    const ElemType newVal = valsVec[ii];
    if (lastVal < ElemType(0) && newVal > ElemType(0))
    {
      assert(padding == 0); // we should arrive here once!
      padding = data.n_elem - data.n_nonzero;
    }
    
    locations.at(0, ii) = (ii + padding) % data.n_rows;
    locations.at(1, ii) = (ii + padding) / data.n_rows;
    vals.at(ii) = lastVal = newVal;
  }
  
  return SpMat<ElemType>(locations, vals, data.n_rows, data.n_cols, false, false);
};


//! @}
