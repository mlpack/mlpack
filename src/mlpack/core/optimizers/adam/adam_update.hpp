/**
 * @file adam_update.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 * @author Marcus Edel
 * @author Vivek Pal
 *
 * Adam optimizer. Adam is an an algorithm for first-order gradient-based
 * optimization of stochastic objective functions, based on adaptive estimates
 * of lower-order moments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADAM_ADAM_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_ADAM_ADAM_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Adam is an optimizer that computes individual adaptive learning rates for
 * different parameters from estimates of first and second moments of the
 * gradients as given in the section 7 of the following paper.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Kingma2014,
 *   author  = {Diederik P. Kingma and Jimmy Ba},
 *   title   = {Adam: {A} Method for Stochastic Optimization},
 *   journal = {CoRR},
 *   year    = {2014},
 *   url     = {http://arxiv.org/abs/1412.6980}
 * }
 * @endcode
 */
class AdamUpdate
{
 public:
  /**
   * Construct the Adam update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  AdamUpdate(const double epsilon = 1e-8,
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

    v *= beta2;
    v += (1 - beta2) * (gradient % gradient);

    const double biasCorrection1 = 1.0 - std::pow(beta1, iteration);
    const double biasCorrection2 = 1.0 - std::pow(beta2, iteration);

    /**
     * It should be noted that the term, m / (arma::sqrt(v) + eps), in the
     * following expression is an approximation of the following actual term;
     * m / (arma::sqrt(v) + (arma::sqrt(biasCorrection2) * eps).
     */
    iterate -= (stepSize * std::sqrt(biasCorrection2) / biasCorrection1) *
        m / (arma::sqrt(v) + epsilon);
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

  // The number of iterations.
  double iteration;
};

} // namespace optimization
} // namespace mlpack

#endif
