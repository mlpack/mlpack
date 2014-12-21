/**
 * @file regularized_svd_function.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of the RegularizedSVDFunction class.
 *
 * This file is part of MLPACK 1.0.9.
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

#ifndef __MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_FUNCTION_SVD_HPP
#define __MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_FUNCTION_SVD_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

namespace mlpack {
namespace svd {

class RegularizedSVDFunction
{
 public:
  
  /**
   * Constructor for RegularizedSVDFunction class. The constructor calculates
   * the number of users and items in the passed data. It also randomly
   * initializes the parameter values.
   *
   * @param data Dataset for which SVD is calculated.
   * @param rank Rank used for matrix factorization.
   * @param lambda Regularization parameter used for optimization.
   */
  RegularizedSVDFunction(const arma::mat& data,
                         const size_t rank,
                         const double lambda);
  
  /**
   * Evaluates the cost function over all examples in the data.
   *
   * @param parameters Parameters(user/item matrices) of the decomposition.
   */
  double Evaluate(const arma::mat& parameters) const;
  
  /**
   * Evaluates the cost function for one training example. Useful for the SGD
   * optimizer abstraction which uses one training example at a time.
   *
   * @param parameters Parameters(user/item matrices) of the decomposition.
   * @param i Index of the training example to be used.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t i) const;
  
  /**
   * Evaluates the full gradient of the cost function over all the training
   * examples.
   *
   * @param parameters Parameters(user/item matrices) of the decomposition.
   * @param gradient Calculated gradient for the parameters.
   */
  void Gradient(const arma::mat& parameters,
                arma::mat& gradient) const;
  
  //! Return the initial point for the optimization.
  const arma::mat& GetInitialPoint() const { return initialPoint; }
  
  //! Return the dataset passed into the constructor.
  const arma::mat& Dataset() const { return data; }
  
  //! Return the number of training examples. Useful for SGD optimizer.
  size_t NumFunctions() const { return data.n_cols; }
  
  //! Return the number of users in the data.
  size_t NumUsers() const { return numUsers; }
  
  //! Return the number of items in the data.
  size_t NumItems() const { return numItems; }
  
  //! Return the regularization parameters.
  double Lambda() const { return lambda; }
  
  //! Return the rank used for the factorization.
  size_t Rank() const { return rank; }
                         
 private:
  //! Rating data.
  const arma::mat& data;
  //! Initial parameter point.
  arma::mat initialPoint;
  //! Rank used for matrix factorization.
  size_t rank;
  //! Regularization parameter for the optimization.
  double lambda;
  //! Number of users in the given dataset.
  size_t numUsers;
  //! Number of items in the given dataset.
  size_t numItems;
};

}; // namespace svd
}; // namespace mlpack

namespace mlpack {
namespace optimization {

  /**
   * Template specialization for SGD optimizer. Used because the gradient
   * affects only a small number of parameters per example, and thus the normal
   * abstraction does not work as fast as we might like it to.
   */
  template<>
  double SGD<mlpack::svd::RegularizedSVDFunction>::Optimize(
      arma::mat& parameters);

}; // namespace optimization
}; // namespace mlpack

#endif
