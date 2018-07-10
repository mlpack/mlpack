/**
 * @file swats_update.hpp
 * @author Marcus Edel
 *
 * SWATS update rule for Switches from Adam to SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SWATS_SWATS_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SWATS_SWATS_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * SWATS is a simple strategy which switches from Adam to SGD when a triggering
 * condition is satisfied. The condition relates to the projection of Adam steps
 * on the gradient subspace.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Keskar2017,
 *   author  = {Nitish Shirish Keskar and Richard Socher},
 *   title   = {Improving Generalization Performance by Switching from Adam to
 *              {SGD}},
 *   journal = {CoRR},
 *   volume  = {abs/1712.07628},
 *   year    = {2017}
 *   url     = {http://arxiv.org/abs/1712.07628},
 * }
 * @endcode
 */
class SWATSUpdate
{
 public:
  /**
   * Construct the SWATS update policy with given parameter.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  SWATSUpdate(const double epsilon = 1e-8,
             const double beta1 = 0.9,
             const double beta2 = 0.999) :
    epsilon(epsilon),
    beta1(beta1),
    beta2(beta2),
    iteration(0),
    phaseSGD(false),
    sgdRate(0),
    sgdLamda(0)
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

    sgdV = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for SWATS.
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

    if (phaseSGD)
    {
      sgdV *= beta1;
      sgdV += gradient;

      iterate -= (1 - beta1) * sgdRate * sgdV;
      return;
    }

    m *= beta1;
    m += (1 - beta1) * gradient;

    v *= beta2;
    v += (1 - beta2) * (gradient % gradient);

    const double biasCorrection1 = 1.0 - std::pow(beta1, iteration);
    const double biasCorrection2 = 1.0 - std::pow(beta2, iteration);

    arma::mat delta = stepSize * m / biasCorrection1 /
        (arma::sqrt(v / biasCorrection2) + epsilon);
    iterate -= delta;

    const double deltaGradient = arma::dot(delta, gradient);
    if (deltaGradient != 0)
    {
      const double rate = arma::dot(delta, delta) / deltaGradient;
      sgdLamda = beta2 * sgdLamda + (1 - beta2) * rate;
      sgdRate = sgdLamda / biasCorrection2;

      if (std::abs(sgdRate - rate) < epsilon && iteration > 1)
      {
        phaseSGD = true;
        sgdV.zeros();
      }
    }
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
  //! The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  //! The smoothing parameter.
  double beta1;

  //! The second moment coefficient.
  double beta2;

  //! The exponential moving average of gradient values.
  arma::mat m;

  //! The exponential moving average of squared gradient values (Adam).
  arma::mat v;

  //! The number of iterations.
  double iteration;

  //! Wether to use the SGD or Adam update rule.
  bool phaseSGD;

  //! The exponential moving average of squared gradient values (SGD).
  arma::mat sgdV;

  //! SGD scaling parameter.
  double sgdRate;

  //! SGD learning rate.
  double sgdLamda;
};

} // namespace optimization
} // namespace mlpack

#endif
