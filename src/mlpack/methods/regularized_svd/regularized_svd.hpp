/**
 * @file regularized_svd.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of Regularized SVD.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
