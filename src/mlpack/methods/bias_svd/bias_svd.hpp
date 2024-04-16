/**
 * @file methods/bias_svd/bias_svd.hpp
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

#include <mlpack/core.hpp>

#include "bias_svd_function.hpp"

namespace mlpack {

/**
 * Bias SVD is an improvement on Regularized SVD which is a matrix
 * factorization techniques. Bias SVD outputs user/item latent vectors
 * and user/item bias, so that \f$ r_{iu} = b_i + b_u + p_i * q_u \f$, where
 * b, p, q are bias, item latent, user latent respectively. Parameters are
 * optmized by Stochastic Gradient Desent(SGD). The updates also penalize the
 * learning of large feature values by means of regularization.
 *
 * An example of how to use the interface is shown below:
 *
 * @code
 * arma::mat data; // Rating data in the form of coordinate list.
 *
 * const size_t rank = 10; // Rank used for the decomposition.
 * const size_t iterations = 10; // Number of iterations used for optimization.
 *
 * const double alpha = 0.005 // Learning rate for the SGD optimizer.
 * const double lambda = 0.02 // Regularization parameter for the optimization.
 *
 * // Make a BiasSVD object.
 * BiasSVD<> biasSVD(iterations, alpha, lambda);
 *
 * arma::mat u, v; // Item and User matrices.
 * arma::vec p, q; // Item and User bias.
 *
 * // Use the Apply() method to get a factorization.
 * rSVD.Apply(data, rank, u, v, p, q);
 * @endcode
 *
 */
template<typename OptimizerType = ens::StandardSGD,
         typename MatType = arma::mat,
         typename VecType = arma::vec>
class BiasSVD
{
 public:
  /**
   * Constructor of Bias SVD. By default SGD optimizer is used in BiasSVD.
   * The optimizer uses a template specialization of Optimize().
   *
   * @param iterations Number of optimization iterations.
   * @param alpha Learning rate for the SGD optimizer.
   * @param lambda Regularization parameter for the optimization.
   */
  BiasSVD(const size_t iterations = 10,
          const double alpha = 0.02,
          const double lambda = 0.05);

  /**
   * Trains the model and obtains user/item matrices and user/item bias.
   *
   * @param data Rating data matrix.
   * @param rank Rank parameter to be used for optimization.
   * @param u Item matrix obtained on decomposition.
   * @param v User matrix obtained on decomposition.
   * @param p Item bias.
   * @param q User bias.
   */
  void Apply(const MatType& data,
             const size_t rank,
             MatType& u,
             MatType& v,
             VecType& p,
             VecType& q);

 private:
  //! Number of optimization iterations.
  size_t iterations;
  //! Learning rate for the SGD optimizer.
  double alpha;
  //! Regularization parameter for the optimization.
  double lambda;
};

} // namespace mlpack

// Include implementation.
#include "bias_svd_impl.hpp"

#endif
