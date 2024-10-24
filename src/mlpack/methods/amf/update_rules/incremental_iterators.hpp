/**
 * @file incremental_iterators.hpp
 * @author Ryan Curtin
 *
 * Implement iterator utilities to iterate over nonzero elements, used by
 * SVDCompleteIncrementalLearning and SVDIncompleteIncrementalLearning.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AMF_UPDATE_RULES_INCREMENTAL_ITERATORS_HPP
#define MLPACK_METHODS_AMF_UPDATE_RULES_INCREMENTAL_ITERATORS_HPP

namespace mlpack {

// Initialize a dense iterator over nonzero values.
template<typename MatType>
void InitializeVIter(const MatType& V,
                     typename MatType::const_iterator& vIter,
                     size_t& currentUserIndex,
                     size_t& currentItemIndex)
{
  vIter = V.begin();
  currentItemIndex = 0;
  currentUserIndex = 0;
  if ((*vIter) == (typename MatType::elem_type) 0.0)
  {
    IncrementVIter(V, vIter, currentUserIndex, currentItemIndex);
  }
}

// Initialize an iterator over values in a sparse matrix.
template<typename eT>
void InitializeVIter(const arma::SpMat<eT>& V,
                     typename arma::SpMat<eT>::const_iterator& vIter,
                     size_t& currentUserIndex,
                     size_t& currentItemIndex)
{
  vIter = V.begin();
  currentItemIndex = vIter.row();
  currentUserIndex = vIter.col();
}

template<typename MatType>
void IncrementVIter(const MatType& V,
                    typename MatType::const_iterator& vIter,
                    size_t& currentUserIndex,
                    size_t& currentItemIndex)
{
  using eT = typename MatType::elem_type;

  // For dense matrices, 0s may be represented, so increment until we find the
  // next nonzero value.
  do
  {
    ++vIter;
    ++currentItemIndex;
    if (currentItemIndex == V.n_rows)
    {
      currentItemIndex = 0;
      currentUserIndex++;
    }

    // If we are past the end of the matrix, go back to the beginning.
    if (currentUserIndex == V.n_cols)
    {
      vIter = V.begin();
      currentUserIndex = 0;
    }
  } while ((*vIter) == (eT) 0);
}

template<typename eT>
void IncrementVIter(const arma::SpMat<eT>& V,
                    typename arma::SpMat<eT>::const_iterator& vIter,
                    size_t& currentUserIndex,
                    size_t& currentItemIndex)
{
  // The iterator automatically handles zero values.
  ++vIter;
  if (vIter == V.end())
  {
    // Reset to start.
    vIter = V.begin();
  }

  currentItemIndex = vIter.row();
  currentUserIndex = vIter.col();
}

} // namespace mlpack

#endif
