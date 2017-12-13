/**
 * @file nadam_update.hpp
 * @author Sourabh Varshney
 *
 * Nadam optimizer. Nadam is an optimizer that combines the effect of Adam and 
 * NAG to the gradient descent to improve its Performance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_NADAM_NADAM_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_NADAM_NADAM_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Nadam is an optimizer that combines the Adam and NAG.
 *
 * For more information, see the following.
 *
 * @code
 * @article{
 *   author  = {Sebastian Ruder},
 *   title   = {An overview of gradient descent optimization algorithms},
 *   journal = {CoRR},
 *   year    = {2016},
 *   url     = {https://arxiv.org/abs/1609.04747v2}
 * }
 * @endcode
 */
class NadamUpdate
{
 public:
  /**
   * Construct the Nadam update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   */
  NadamUpdate(const double epsilon = 1e-8, const double beta1 = 0.9)
  :epsilon(epsilon), beta1(beta1), iteration(0)
  {
    // Nothing to do.
  }

  /**
   * The Initialize method is called by SGD Optimizer method before the start 
   * of the iteration update process.
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
   * Update step for Nadam.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate, const double stepSize,
  const arma::mat& gradient)
  {
    // Increment the iteration counter variable.
    ++iteration;

    // And update the iterate.
    m *= beta1;
    m += (1 - beta1) * gradient;
    // biasCorrection=1-beta1^iteration
    const double biasCorrection = 1.0 - std::pow(beta1, iteration);
    /*
	iterate=iterate-((stepsize/(sqrt(v)+epsilon))*(m/biasCorrection))
    */
    iterate -= ((stepSize * m)/(biasCorrection1 * (arma::sqrt(v) + epsilon)));
  }

  //! Get the value used to initialise the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the squared gradient parameter.
  double& Epsilon() { return epsilon; }

  //! Get the smoothing parameter.
  double Beta1() const { return beta1; }
  //! Modify the smoothing parameter.
  double& Beta1() { return beta1; }

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double beta1;

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
