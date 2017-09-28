/**
 * @file rmsprop_update.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 * @author Vivek Pal
 *
 * RMSProp optimizer. RMSProp is an optimizer that utilizes the magnitude of
 * recent gradients to normalize the gradients.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_RMSPROP_RMSPROP_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_RMSPROP_RMSPROP_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * RMSProp is an optimizer that utilizes the magnitude of recent gradients to
 * normalize the gradients. In its basic form, given a step rate \f$ \gamma \f$
 * and a decay term \f$ \alpha \f$ we perform the following updates:
 *
 * \f{eqnarray*}{
 * r_t &=& (1 - \gamma) f'(\Delta_t)^2 + \gamma r_{t - 1} \\
 * v_{t + 1} &=& \frac{\alpha}{\sqrt{r_t}}f'(\Delta_t) \\
 * \Delta_{t + 1} &=& \Delta_t - v_{t + 1}
 * \f}
 *
 * For more information, see the following.
 *
 * @code
 * @misc{tieleman2012,
 *   title = {Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine
 *            Learning},
 *   year  = {2012}
 * }
 * @endcode
 */
class RMSPropUpdate
{
 public:
  /**
   * Construct the RMSProp update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param alpha The smoothing parameter.
   */
  RMSPropUpdate(const double epsilon = 1e-8,
                const double alpha = 0.99) :
    epsilon(epsilon),
    alpha(alpha)
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
    // Leaky sum of squares of parameter gradient.
    meanSquaredGradient = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for RMSProp.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    meanSquaredGradient *= alpha;
    meanSquaredGradient += (1 - alpha) * (gradient % gradient);
    iterate -= stepSize * gradient / (arma::sqrt(meanSquaredGradient) +
        epsilon);
  }

  //! Get the value used to initialise the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the squared gradient parameter.
  double& Epsilon() { return epsilon; }

  //! Get the smoothing parameter.
  double Alpha() const { return alpha; }
  //! Modify the smoothing parameter.
  double& Alpha() { return alpha; }

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double alpha;

  // Leaky sum of squares of parameter gradient.
  arma::mat meanSquaredGradient;
};

} // namespace optimization
} // namespace mlpack

#endif
