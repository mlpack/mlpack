/**
 * @file empty_update.hpp
 * @author Arun Reddy
 *
 * Empty update for SGD
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
 * Empty update policy for SGD.
 *
 */
template<class UpdatePolicyType>
class MomentumUpdate {
 public:

  /**
   * Momentum update constructor.
   *
   * @param momentum The momentum hyperparameter
   */
  MomentumUpdate(const double momentum = 0.5);

  void Initialize(const size_t n_rows = 0,
            const size_t n_cols = 0)
  {
    //Initialize am empty velocity matrix.
    velocity(arma::zeros<arma::mat>(n_rows, n_cols));
  }


  arma::mat Update(const double stepSize,
              arma::mat& gradient)
  {
    velocity = mu*velocity - stepSize * gradient;
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
