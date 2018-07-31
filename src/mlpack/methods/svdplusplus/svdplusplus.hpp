/**
 * @file svdplusplus.hpp
 * @author Siddharth Agrawal
 * @author Wenhao Huang
 *
 * An implementation of SVD++.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_HPP
#define MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/cf/cf.hpp>

#include "svdplusplus_function.hpp"

namespace mlpack {
namespace svd {

/**
 *
 */
template<typename OptimizerType = mlpack::optimization::StandardSGD>
class SVDPlusPlus
{
 public:
  /**
   * 
   * @param iterations Number of optimization iterations.
   * @param alpha Learning rate for the SGD optimizer.
   * @param lambda Regularization parameter for the optimization.
   */
  SVDPlusPlus(const size_t iterations = 10,
          const double alpha = 0.001,
          const double lambda = 0.1);

  /**
   *
   * @param data Rating data matrix.
   * @param implicitData Implicit feedback.
   * @param rank Rank parameter to be used for optimization.
   * @param u Item matrix obtained on decomposition.
   * @param v User matrix obtained on decomposition.
   * @param p Item bias.
   * @param q User bias.
   * @param y Item matrix with respect to implicit feedback.
   */
  void Apply(const arma::mat& data,
             const arma::mat& implicitData,
             const size_t rank,
             arma::mat& u,
             arma::mat& v,
             arma::vec& p,
             arma::vec& q,
             arma::mat& y);
  
  /**
   * Whether the user rates an item is used as implicit feedback.
   *
   * @param data Rating data matrix.
   * @param rank Rank parameter to be used for optimization.
   * @param u Item matrix obtained on decomposition.
   * @param v User matrix obtained on decomposition.
   * @param p Item bias.
   * @param q User bias.
   * @param y Item matrix with respect to implicit feedback. Each column is a
   *     latent vector of an item with respect to implicit feedback.
   */
  void Apply(const arma::mat& data,
             const size_t rank,
             arma::mat& u,
             arma::mat& v,
             arma::vec& p,
             arma::vec& q,
             arma::mat& y);
  
  /**
   * Converts the User, Item matrix of implicit data to Item-User Table.
   */
  static void CleanData(const arma::mat& implicitData,
                        arma::sp_mat& cleanedData,
                        const arma::mat& data);

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
#include "svdplusplus_impl.hpp"

#endif
