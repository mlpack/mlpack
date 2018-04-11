/**
 * @file sga_update.hpp
 * @author Rohan Raj
 *
 * Update for Stochastic Gradient Ascent for Reinforcement Learning.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGA_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SGA_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Update policy for Stochastic Gradient Ascent (SGA).
*/
class SgaUpdate
{
 public:
  /**
   * The Initialize method may be called by any Optimizer method before the 
   * start of the iteration update process.  This update doesn't initialize
   * anything. Since gradient ascent is generally used in Policy gradient
   * methods in reinforcement learning, it can be directly called by the 
   * agents to update its policy network via gradient ascent.
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t /* rows */, const size_t /* cols */)
  { /* Do nothing. */ }

 /**
  * Update step for SGA.  The function parameters are updated in the
  * direction of the gradients.
  *
  * @param iterate Parameters that maximize the reward for agent.
  * @param stepSize Step size to be used for the given iteration.
  * @param gradient The gradient matrix.
  */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    // Perform the SGA update.
    iterate += stepSize * gradient;
  }
};

} // namespace optimization
} // namespace mlpack

#endif
