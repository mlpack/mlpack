/**
 * @file nadamax_update.hpp
 * @author Sourabh Varshney
 *
 * NadaMax update rule. NadaMax is an optimizer that combines the effect of
 * Adamax and NAG to the gradient descent to improve its Performance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADAM_NADAMAX_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_ADAM_NADAMAX_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * NadaMax is an optimizer that combines the AdaMax and NAG.
 *
 * For more information, see the following.
 *
 * @code
 * @techreport{Dozat2015,
 *   title       = {Incorporating Nesterov momentum into Adam},
 *   author      = {Timothy Dozat},
 *   institution = {Stanford University},
 *   address     = {Stanford},
 *   year        = {2015},
 *   url         = {https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ}
 * }
 * @endcode
 */
class NadaMaxUpdate
{
 public:
  /**
   * Construct the NadaMax update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient
   * @param scheduleDecay The decay parameter for decay coefficients
   */
  NadaMaxUpdate(const double epsilon = 1e-8,
                const double beta1 = 0.9,
                const double beta2 = 0.99,
                const double scheduleDecay = 4e-3) :
      epsilon(epsilon),
      beta1(beta1),
      beta2(beta2),
      scheduleDecay(scheduleDecay),
      cumBeta1(1),
      iteration(0)
  {
    // Nothing to do.
  }

  /**
   * The Initialize() method is called by the optimizer before the start of the
   * iteration update process.
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
   * Update step for NadaMax.
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

    u = arma::max(u * beta2, arma::abs(gradient));

    double beta1T = beta1 * (1 - (0.5 *
        std::pow(0.96, iteration * scheduleDecay)));

    double beta1T1 = beta1 * (1 - (0.5 *
        std::pow(0.96, (iteration + 1) * scheduleDecay)));

    cumBeta1 *= beta1T;

    const double biasCorrection1 = 1.0 - cumBeta1;

    const double biasCorrection2 = 1.0 - (cumBeta1 * beta1T1);

    if ((biasCorrection1 != 0) && (biasCorrection2 != 0))
    {
       iterate -= (stepSize * (((1 - beta1T) / biasCorrection1) * gradient
           + (beta1T1 / biasCorrection2) * m)) / (u + epsilon);
    }
  }

  //! Get the value used to initialise the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the squared gradient parameter.
  double& Epsilon() { return epsilon; }

  //! Get the value of the cumulative product of decay coefficients
  double CumBeta1() const { return cumBeta1; }
  //! Modify the value of the cumulative product of decay coefficients
  double& CumBeta1() { return cumBeta1; }

  //! Get the smoothing parameter.
  double Beta1() const { return beta1; }
  //! Modify the smoothing parameter.
  double& Beta1() { return beta1; }

  //! Get the second moment coefficient.
  double Beta2() const { return beta2; }
  //! Modify the second moment coefficient.
  double& Beta2() { return beta2; }

  //! Get the decay parameter for decay coefficients
  double ScheduleDecay() const { return scheduleDecay; }
  //! Modify the decay parameter for decay coefficients
  double& ScheduleDecay() { return scheduleDecay; }

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

  // The decay parameter for decay coefficients
  double scheduleDecay;

  // The cumulative product of decay coefficients
  double cumBeta1;

  // The number of iterations.
  double iteration;
};

} // namespace optimization
} // namespace mlpack

#endif
