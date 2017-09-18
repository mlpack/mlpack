/**
 * @file ada_delta_update.hpp
 * @author Vasanth Kalingeri
 * @author Abhinav Moudgil
 *
 * AdaDelta update for Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADA_DELTA_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_ADA_DELTA_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Implementation of the AdaDelta update policy. AdaDelta is an optimizer that
 * uses two ideas to improve upon the two main drawbacks of the AdaGrad method:
 *
 * - Accumulate Over Window
 * - Correct Units with Hessian Approximation
 *
 * For more information, see the following.
 *
 * @code
 * @article{Zeiler2012,
 *   author  = {Matthew D. Zeiler},
 *   title   = {{ADADELTA:} An Adaptive Learning Rate Method},
 *   journal = {CoRR},
 *   year    = {2012}
 * }
 * @endcode
 *
 */
class AdaDeltaUpdate
{
 public:
  /**
   * Construct the AdaDelta update policy with given rho and epsilon parameters.
   *
   * @param rho The smoothing parameter.
   * @param epsilon The epsilon value used to initialise the squared gradient
   *    parameter.
   */
  AdaDeltaUpdate(const double rho = 0.95, const double epsilon = 1e-6) :
      rho(rho),
      epsilon(epsilon)
  {
    // Nothing to do.
  }

  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process. In AdaDelta update policy, the mean squared
   * and the delta mean squared gradient matrices are initialized to the zeros
   * matrix with the same size as gradient matrix
   * (see mlpack::optimization::SGD::Optimizer).
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t rows, const size_t cols)
  {
    // Initialize empty matrices for mean sum of squares of parameter gradient.
    meanSquaredGradient = arma::zeros<arma::mat>(rows, cols);
    meanSquaredGradientDx = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for SGD. The AdaDelta update dynamically adapts over time using
   * only first order information. Additionally, AdaDelta requires no manual
   * tuning of a learning rate.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    // Accumulate gradient.
    meanSquaredGradient *= rho;
    meanSquaredGradient += (1 - rho) * (gradient % gradient);
    arma::mat dx = arma::sqrt((meanSquaredGradientDx + epsilon) /
        (meanSquaredGradient + epsilon)) % gradient;

    // Accumulate updates.
    meanSquaredGradientDx *= rho;
    meanSquaredGradientDx += (1 - rho) * (dx % dx);

    // Apply update.
    iterate -= (stepSize * dx);
  }

  //! Get the smoothing parameter.
  double Rho() const { return rho; }
  //! Modify the smoothing parameter.
  double& Rho() { return rho; }

  //! Get the value used to initialise the mean squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the mean squared gradient parameter.
  double& Epsilon() { return epsilon; }

 private:
  // The smoothing parameter.
  double rho;

  // The epsilon value used to initialise the mean squared gradient parameter.
  double epsilon;

  // The mean squared gradient matrix.
  arma::mat meanSquaredGradient;

  // The delta mean squared gradient matrix.
  arma::mat meanSquaredGradientDx;
};

} // namespace optimization
} // namespace mlpack

#endif
