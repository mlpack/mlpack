/**
 * @file optimisticadam_update.hpp
 * @author Moksh Jain
 *
 * OptmisticAdam optimizer. Implements Optimistic Adam, an algorithm which 
 * uses Optimistic Mirror Descent with the Adam optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADAM_OPTIMISTICADAM_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_ADAM_OPTIMISTICADAM_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * OptimisticAdam is an optimizer which implements the Optimistic Adam 
 * algorithm which uses Optmistic Mirror Descent with the Adam Optimizer.
 * It addresses the problem of limit cycling while training GANs. It uses
 * OMD to achieve faster regret rates in solving the zero sum game of 
 * training a GAN. It consistently achieves a smaller KL divergnce with 
 * respect to the true underlying data distribution.
 *
 * For more information, see the following.
 *
 * @code
 * @article{
 *   author  = {Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis, 
 *              Haoyang Zeng},
 *   title   = {Training GANs with Optimism},
 *   year    = {2017},
 *   url     = {https://arxiv.org/abs/1711.00141}
 * }
 * @endcode
 */
class OptimisticAdamUpdate
{
 public:
  /**
   * Construct the OptimisticAdam update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialize the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  OptimisticAdamUpdate(const double epsilon = 1e-8,
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
    g = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for OptimisticAdam.
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
    v += (1 - beta2) * arma::square(gradient);

    arma::mat mCorrected = m / (1.0 - std::pow(beta1, iteration));
    arma::mat vCorrected = v / (1.0 - std::pow(beta2, iteration));

    arma::mat update = mCorrected / (arma::sqrt(vCorrected) + epsilon);

    iterate -= (2 * stepSize * update - stepSize * g);

    g = std::move(update);
  }

  //! Get the value used to initialize the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialize the squared gradient parameter.
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
  // The epsilon value used to initialize the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double beta1;

  // The second moment coefficient.
  double beta2;

  // The exponential moving average of gradient values.
  arma::mat m;

  // The exponential moving average of squared gradient values.
  arma::mat v;
  // The previous update.
  arma::mat g;

  // The number of iterations.
  double iteration;
};

} // namespace optimization
} // namespace mlpack

#endif
