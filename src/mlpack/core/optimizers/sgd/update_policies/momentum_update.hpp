/**
 * @file momentum_update.hpp
 * @author Arun Reddy
 *
 * Momentum update for SGD
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGD_MOMENTUM_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_MOMENTUM_UPDATE_HPP

namespace mlpack {
namespace optimization {

/**
 * Momentum update policy for SGD.
 *
 */
class MomentumUpdate {
 public:

  /**
   * Momentum update constructor.
   *
   * @param momentum The momentum hyperparameter
   */
  MomentumUpdate(const double momentum = 0.5):
      momentum(momentum)
  {/* do nothing */};

  /**
   * Intialize the velocity matrix to the zeros matrix of the given size
   * @param n_rows number of rows in the gradient matrix.
   * @param n_cols number of columns in the gradient matrix.
   */
  void Initialize(const size_t n_rows,
                  const size_t n_cols)
  {
    //Initialize am empty velocity matrix.
    velocity = arma::zeros<arma::mat>(n_rows, n_cols);
  }

  /**
   * Gradient update step for SGD.
   * @param stepSize
   * @param gradient The gradient matrix.
   * @return Updated gradient matrix.
   */
  arma::mat Update(const double stepSize,
                   arma::mat gradient)
  {
    velocity = momentum*velocity - stepSize * gradient;
    return velocity;
  }

 private:

  // The momentum hyperparamter
  double momentum;

  // The velocity matrix.
  arma::mat velocity;

};

} // namespace optimization
} // namespace mlpack

#endif
