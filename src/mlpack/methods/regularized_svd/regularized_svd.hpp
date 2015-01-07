/**
 * @file regularized_svd.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of Regularized SVD.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef __MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_HPP
#define __MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

#include "regularized_svd_function.hpp"

namespace mlpack {
namespace svd {

template<
  template<typename> class OptimizerType = mlpack::optimization::SGD
>
class RegularizedSVD
{
 public:
 
  /**
   * Constructor for Regularized SVD. Obtains the user and item matrices after
   * training on the passed data. The constructor initiates an object of class
   * RegularizedSVDFunction for optimization. It uses the SGD optimizer by
   * default. The optimizer uses a template specialization of Optimize().
   *
   * @param data Dataset for which SVD is calculated.
   * @param u User matrix in the matrix decomposition.
   * @param v Item matrix in the matrix decomposition.
   * @param rank Rank used for matrix factorization.
   * @param iterations Number of optimization iterations.
   * @param lambda Regularization parameter for the optimization.
   */
  RegularizedSVD(const arma::mat& data,
                 arma::mat& u,
                 arma::mat& v,
                 const size_t rank,
                 const size_t iterations = 10,
                 const double alpha = 0.01,
                 const double lambda = 0.02);
                 
 private:
  //! Rating data.
  const arma::mat& data;
  //! Rank used for matrix factorization.
  size_t rank;
  //! Number of optimization iterations.
  size_t iterations;
  //! Learning rate for the SGD optimizer.
  double alpha;
  //! Regularization parameter for the optimization.
  double lambda;
  //! Function that will be held by the optimizer.
  RegularizedSVDFunction rSVDFunc;
  //! Default SGD optimizer for the class.
  mlpack::optimization::SGD<RegularizedSVDFunction> optimizer;
};

}; // namespace svd
}; // namespace mlpack

// Include implementation.
#include "regularized_svd_impl.hpp"

#endif
