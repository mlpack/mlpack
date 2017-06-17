/**
 * @file vanilla_update.hpp
 * @author Arun Reddy
 *
 * Vanilla update for Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGD_EMPTY_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_EMPTY_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Vanilla update policy for Stochastic Gradient Descent (SGD). The following
 * update scheme is used to update SGD in every iteration:
 *
 * \f[
 * A_{j + 1} = A_j + \alpha \nabla f_i(A)
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size.  \f$ i \f$
 * is chosen according to \f$ j \f$ (the iteration number).
 */
class VanillaUpdate
{
 public:
  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process.  The vanilla update doesn't initialize
   * anything.
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t /* rows */, const size_t /* cols */)
  { /* Do nothing. */ }

 /**
  * Update step for SGD.  The function parameters are updated in the negative
  * direction of the gradient.
  *
  * @param iterate Parameters that minimize the function.
  * @param stepSize Step size to be used for the given iteration.
  * @param gradient The gradient matrix.
  */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    // Perform the vanilla SGD update.
    iterate -= stepSize * gradient;
  }
};

} // namespace optimization
} // namespace mlpack

#endif
