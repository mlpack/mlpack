/**
 * @file bias_svd.hpp
 * @author Siddharth Agrawal
 * @author Wenhao Huang
 *
 * An implementation of Bias SVD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_BIAS_SVD_BIAS_SVD_HPP
#define MLPACK_METHODS_BIAS_SVD_BIAS_SVD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/cf/cf.hpp>

#include "bias_svd_function.hpp"

namespace mlpack {
namespace svd {

/**
 *
 */
template<typename OptimizerType = mlpack::optimization::StandardSGD>
class BiasSVD
{
 public:
  /**
   * 
   * @param iterations Number of optimization iterations.
   * @param alpha Learning rate for the SGD optimizer.
   * @param lambda Regularization parameter for the optimization.
   */
  BiasSVD(const size_t iterations = 10,
          const double alpha = 0.005,
          const double lambda = 0.02);

  /**
   *
   * @param data Rating data matrix.
   * @param rank Rank parameter to be used for optimization.
   * @param u Item matrix obtained on decomposition.
   * @param v User matrix obtained on decomposition.
   * @param p Item bias.
   * @param q User bias.
   */
  void Apply(const arma::mat& data,
             const size_t rank,
             arma::mat& u,
             arma::mat& v,
             arma::vec& p,
             arma::vec& q);

 private:
  //! Number of optimization iterations.
  size_t iterations;
  //! Learning rate for the SGD optimizer.
  double alpha;
  //! Regularization parameter for the optimization.
  double lambda;
};

} // namespace svd
} // namespace mlpack

// Include implementation.
#include "bias_svd_impl.hpp"

#endif
