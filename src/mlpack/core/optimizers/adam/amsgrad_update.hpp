/**
 * @file amsgrad_update.hpp
 * @author Haritha Nair
 *
 * Implementation of AMSGrad optimizer. AMSGrad is an exponential moving average 
 * optimizer that dynamically adapts over time with guaranteed convergence.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_AMS_GRAD_AMS_GRAD_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_AMS_GRAD_AMS_GRAD_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * AMSGrad is an exponential moving average variant which along with having
 * benefits of optimizers like Adam and RMSProp, also guarantees convergence.
 * Unlike Adam, it uses maximum of past squared gradients rather than their
 * exponential average for updation.
 *
 * For more information, see the following.
 *
 * @code
 * @article{
 *   title   = {On the convergence of Adam and beyond},
 *   url     = {https://openreview.net/pdf?id=ryQu7f-RZ}
 *   year    = {2018}
 * }
 * @endcode
 */
class AMSGradUpdate
{
 public:
  /**
   * Construct the AMSGrad update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  AMSGradUpdate(const double epsilon = 1e-8,
                const double beta1 = 0.9,
                const double beta2 = 0.999) :
    epsilon(epsilon),
    beta1(beta1),
    beta2(beta2),
    iteration(0)
  {
    // Nothing to do.
  }

  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process.
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t rows, const size_t cols)
  {
    m = arma::zeros<arma::mat>(rows, cols);
    v = arma::zeros<arma::mat>(rows, cols);
    vImproved = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for AMSGrad.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    // Increment the iteration counter variable.
    ++iteration;

    // And update the iterate.
    m *= beta1;
    m += (1 - beta1) * gradient;

    v *= beta2;
    v += (1 - beta2) * (gradient % gradient);

    const double biasCorrection1 = 1.0 - std::pow(beta1, iteration);
    const double biasCorrection2 = 1.0 - std::pow(beta2, iteration);

    // Element wise maximum of past and present squared gradients.
    vImproved = arma::max(vImproved, v);

    iterate -= (stepSize * std::sqrt(biasCorrection2) / biasCorrection1) *
                m / (arma::sqrt(vImproved) + epsilon);
  }

  //! Get the value used to initialise the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the squared gradient parameter.
  double& Epsilon() { return epsilon; }

  //! Get the smoothing parameter.
  double Beta1() const { return beta1; }
  //! Modify the smoothing parameter.
  double& Beta1() { return beta1; }

  //! Get the second moment coefficient.
  double Beta2() const { return beta2; }
  //! Modify the second moment coefficient.
  double& Beta2() { return beta2; }

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double beta1;

  // The second moment coefficient.
  double beta2;

  // The exponential moving average of gradient values.
  arma::mat m;

  // The exponential moving average of squared gradient values.
  arma::mat v;

  // The optimal sqaured gradient value.
  arma::mat vImproved;

  // The number of iterations.
  double iteration;
};

} // namespace optimization
} // namespace mlpack

#endif
