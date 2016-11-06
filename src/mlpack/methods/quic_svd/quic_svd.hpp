/**
 * @file quic_svd.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of QUIC-SVD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_QUIC_SVD_QUIC_SVD_HPP
#define MLPACK_METHODS_QUIC_SVD_QUIC_SVD_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cosine_tree/cosine_tree.hpp>

namespace mlpack {
namespace svd {

/**
 * QUIC-SVD is a matrix factorization technique, which operates in a subspace
 * such that A's approximation in that subspace has minimum error(A being the
 * data matrix). The subspace is constructed using a cosine tree, which ensures
 * minimum representative rank(and thus a fast running time). It follows a
 * splitting policy based on Length-squared(LS) sampling and constructs the
 * child nodes based on the absolute cosines of the remaining points relative to
 * the pivot. The centroids of the points in the child nodes are added to the
 * subspace span in each step. Each node is then placed into a queue prioritized
 * by its residual error. The subspace approximation error of A after each step
 * is calculated using a Monte Carlo estimate. If the error is below a certain
 * threshold, the method proceeds to calculate the Singular Value Decomposition
 * in the obtained subspace. Otherwise, the same procedure is repeated until we
 * obtain a subspace of sufficiently low error. Technical details can be found
 * in the following paper:
 *
 * http://www.cc.gatech.edu/~isbell/papers/isbell-quicsvd-nips-2008.pdf
 *
 * An example of how to use the interface is shown below:
 *
 * @code
 * arma::mat data; // Data matrix.
 *
 * const double epsilon = 0.01; // Relative error limit of data in subspace.
 * const double delta = 0.1 // Lower error bound for Monte Carlo estimate.
 *
 * arma::mat u, v, sigma; // Matrices for the factors. data = u * sigma * v.t()
 *
 * // Get the factorization in the constructor.
 * QUIC_SVD(data, u, v, sigma, epsilon, delta);
 * @endcode
 */
class QUIC_SVD
{
 public:
  /**
   * Constructor which implements the QUIC-SVD algorithm. The function calls the
   * CosineTree constructor to create a subspace basis, where the original
   * matrix's projection has minimum reconstruction error. The constructor then
   * uses the ExtractSVD() function to calculate the SVD of the original dataset
   * in that subspace.
   *
   * @param dataset Matrix for which SVD is calculated.
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param sigma Diagonal matrix of singular values.
   * @param epsilon Error tolerance fraction for calculated subspace.
   * @param delta Cumulative probability for Monte Carlo error lower bound.
   */
  QUIC_SVD(const arma::mat& dataset,
           arma::mat& u,
           arma::mat& v,
           arma::mat& sigma,
           const double epsilon = 0.03,
           const double delta = 0.1);

  /**
   * This function uses the vector subspace created using a cosine tree to
   * calculate an approximate SVD of the original matrix.
   *
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param sigma Diagonal matrix of singular values.
   */
  void ExtractSVD(arma::mat& u, arma::mat& v, arma::mat& sigma);

 private:
  //! Matrix for which cosine tree is constructed.
  const arma::mat& dataset;
  //! Subspace basis of the input dataset.
  arma::mat basis;
};

} // namespace svd
} // namespace mlpack

#endif
