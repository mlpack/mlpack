/**
 * @file nesterov_momentum_update.hpp
 * @author Sourabh Varshney
 *
 * Nesterov Momentum Update for Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGD_NESTEROV_MOMENTUM_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_NESTEROV_MOMENTUM_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Nesterov Momentum update policy for Stochastic Gradient Descent (SGD).
 *
 * Learning with SGD can be slow. Applying Standard momentum can accelerate
 * the rate of convergence. Nesterov Momentum application can accelerate the
 * rate of convergence to O(1/k^2).
 * 
 * @code
 * @techreport{Nesterov1983,
 *   title       = {A Method Of Solving A Convex Programming Problem With 
 *                  Convergence Rate O(1/K^2)},
 *   author      = {Yuri Nesterov},
 *   institution = {Soviet Math. Dokl.},
 *   volume      = {27},
 *   year        = {1983},
 * }
 * @endcode
 */
class NesterovMomentumUpdate
{
 public:
  /**
   * Construct the Nesterov Momentum update policy with the given parameters.
   *
   */
  NesterovMomentumUpdate(const double momentum = 0.5) :
      momentum(momentum)
  {
    // Nothing to do.
  }

  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process.  In the momentum update policy the velocity
   * matrix is initialized to the zeros matrix with the same size as the
   * gradient matrix (see mlpack::optimization::SGD::Optimizer )
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t rows, const size_t cols)
  {
    // Initialize an empty velocity matrix.
    velocity = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for SGD.  The momentum term makes the convergence faster on the
   * way as momentum term increases for dimensions pointing in the same direction
   * and reduces updates for dimensions whose gradients change directions.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    velocity = momentum * velocity - stepSize * gradient;

    iterate += momentum * velocity - stepSize * gradient;
  }

  //! Get the value used to initialize the momentum coefficient.
  double Momentum() const { return momentum; }
  //! Modify the value used to initialize the momentum coefficient.
  double& Momentum() { return momentum; }

 private:
  // The velocity matrix.
  arma::mat velocity;

  // The Momentum coefficient.
  double momentum;
};

} // namespace optimization
} // namespace mlpack

#endif
