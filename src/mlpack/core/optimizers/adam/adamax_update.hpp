/**
 * @file adamax_update.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 * @author Marcus Edel
 * @author Vivek Pal
 *
 * AdaMax update rule. Adam is an an algorithm for first-order gradient-
 * -based optimization of stochastic objective functions, based on adaptive
 * estimates of lower-order moments. AdaMax is simply a variant of Adam based
 * on the infinity norm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADAM_ADAMAX_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_ADAM_ADAMAX_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * AdaMax is a variant of Adam, an optimizer that computes individual adaptive
 * learning rates for different parameters from estimates of first and second
 * moments of the gradients.based on the infinity norm as given in the section
 * 7 of the following paper.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Kingma2014,
 *   author    = {Diederik P. Kingma and Jimmy Ba},
 *   title     = {Adam: {A} Method for Stochastic Optimization},
 *   journal   = {CoRR},
 *   year      = {2014},
 *   url       = {http://arxiv.org/abs/1412.6980}
 * }
 * @endcode
 */
class AdaMaxUpdate
{
 public:
  /**
   * Construct the AdaMax update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  AdaMaxUpdate(const double epsilon = 1e-8,
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
    u = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for Adam.
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

    // Update the exponentially weighted infinity norm.
    u *= beta2;
    u = arma::max(u, arma::abs(gradient));

    const double biasCorrection1 = 1.0 - std::pow(beta1, iteration);

    if (biasCorrection1 != 0)
      iterate -= (stepSize / biasCorrection1 * m / (u + epsilon));
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

  // The exponentially weighted infinity norm.
  arma::mat u;

  // The number of iterations.
  double iteration;
};

} // namespace optimization
} // namespace mlpack

#endif
