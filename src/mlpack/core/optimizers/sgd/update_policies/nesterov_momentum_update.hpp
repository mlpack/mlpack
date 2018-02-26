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

/*
 */

class NesterovMomentumUpdate
{
 public:
  /*
  */
  NesterovMomentumUpdate(const double beta1 = 0.99 ,
              		     const double scheduleDecay = 4e-3) :
      beta1(beta1),
      scheduleDecay(scheduleDecay),
      iteration(0)
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
    // Initialize am empty velocity matrix.
    velocity = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for SGD.  The momentum term makes the convergence faster on the
   * way as momentum term increases for dimensions pointing in the same and
   * reduces updates for dimensions whose gradients change directions.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
	double beta1T = beta1 * (1 - (0.5 *
        std::pow(0.96, (iteration - 1) * scheduleDecay)));

    double beta1T1 = beta1 * (1 - (0.5 *
        std::pow(0.96, iteration * scheduleDecay)));

	iterate = iterate + (beta1T * beta1T1 * velocity) - ((1 + beta1T1) * stepSize * gradient);
    
	velocity = beta1T * velocity - stepSize * gradient;
  }

  //! Get the smoothing parameter.
  double Beta1() const { return beta1; }
  //! Modify the smoothing parameter.
  double& Beta1() { return beta1; }

  //! Get the decay parameter for decay coefficients
  double ScheduleDecay() const { return scheduleDecay; }
  //! Modify the decay parameter for decay coefficients
  double& ScheduleDecay() { return scheduleDecay; }

 private:
  // The smoothing parameter.
  double beta1;

  // The velocity matrix.
  arma::mat velocity;

};

} // namespace optimization
} // namespace mlpack

#endif
