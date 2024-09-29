/**
 * @file methods/pca/decomposition_policies/exact_svd_method.hpp
 * @author Ajinkya Kale
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the exact svd method for use in the Principal Components
 * Analysis method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_EXACT_SVD_METHOD_HPP
#define MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_EXACT_SVD_METHOD_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of the exact SVD policy.
 */
class ExactSVDPolicy
{
 public:
  /**
   * Apply Principal Component Analysis to the provided data set using the
   * exact SVD method.
   *
   * @param data Data matrix.
   * @param centeredData Centered data matrix.
   * @param transformedData Matrix to put results of PCA into.
   * @param eigVal Vector to put eigenvalues into.
   * @param eigvec Matrix to put eigenvectors (loadings) into.
   * @param * (rank) Rank of the decomposition.
   */
  template<typename InMatType, typename MatType, typename VecType>
  void Apply(const InMatType& /* data */,
             const MatType& centeredData,
             MatType& transformedData,
             VecType& eigVal,
             MatType& eigvec,
             const size_t /* rank */)
  {
    // This matrix will store the right singular vectors; we do not need them.
    MatType v;

    // Do singular value decomposition.  Use the economical singular value
    // decomposition if the columns are much larger than the rows.
    if (centeredData.n_rows < centeredData.n_cols)
    {
      // Do economical singular value decomposition and compute only the left
      // singular vectors.
      arma::svd_econ(eigvec, eigVal, v, centeredData, 'l');
    }
    else
    {
      arma::svd(eigvec, eigVal, v, centeredData);
    }

    // Now we must square the singular values to get the eigenvalues.
    // In addition we must divide by the number of points, because the
    // covariance matrix is X * X' / (N - 1).
    eigVal %= eigVal / (centeredData.n_cols - 1);

    // Project the samples to the principals.
    transformedData = trans(eigvec) * centeredData;
  }
};

} // namespace mlpack

#endif
