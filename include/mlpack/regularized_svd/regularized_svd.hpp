/**
 * @file methods/regularized_svd/regularized_svd.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of Regularized SVD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_HPP
#define MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_HPP

#include <mlpack/core.hpp>

#include "regularized_svd_function.hpp"

namespace mlpack {

/**
 * Regularized SVD is a matrix factorization technique that seeks to reduce the
 * error on the training set, that is on the examples for which the ratings have
 * been provided by the users. It is a fairly straightforward technique where
 * the user and item matrices are updated with the help of Stochastic Gradient
 * Descent(SGD) updates. The updates also penalize the learning of large feature
 * values by means of regularization. More details can be found in the following
 * links:
 *
 * http://sifter.org/~simon/journal/20061211.html
 * http://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf
 *
 * An example of how to use the interface is shown below:
 *
 * @code
 * arma::mat data; // Rating data in the form of coordinate list.
 *
 * const size_t rank = 20; // Rank used for the decomposition.
 * const size_t iterations = 10; // Number of iterations used for optimization.
 *
 * const double alpha = 0.01 // Learning rate for the SGD optimizer.
 * const double lambda = 0.1 // Regularization parameter for the optimization.
 *
 * // Make a RegularizedSVD object.
 * RegularizedSVD<> rSVD(iterations, alpha, lambda);
 *
 * arma::mat u, v; // User and item matrices.
 *
 * // Use the Apply() method to get a factorization.
 * rSVD.Apply(data, rank, u, v);
 * @endcode
 */
template<typename OptimizerType = ens::StandardSGD>
class RegularizedSVD
{
 public:
  /**
   * Constructor for Regularized SVD. Obtains the user and item matrices after
   * training on the passed data. The constructor initiates an object of class
   * RegularizedSVDFunction for optimization. It uses the SGD optimizer by
   * default. The optimizer uses a template specialization of Optimize().
   *
   * @param iterations Number of optimization iterations.
   * @param alpha Learning rate for the SGD optimizer.
   * @param lambda Regularization parameter for the optimization.
   */
  RegularizedSVD(const size_t iterations = 10,
                 const double alpha = 0.01,
                 const double lambda = 0.02);

  /**
   * Obtains the user and item matrices using the provided data and rank.
   *
   * @param data Rating data matrix.
   * @param rank Rank parameter to be used for optimization.
   * @param u Item matrix obtained on decomposition.
   * @param v User matrix obtained on decomposition.
   */
  void Apply(const arma::mat& data,
             const size_t rank,
             arma::mat& u,
             arma::mat& v);

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
#include "regularized_svd_impl.hpp"

#endif
