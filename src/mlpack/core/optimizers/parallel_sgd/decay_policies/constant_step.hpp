/**
 * @file constant_step.hpp
 * @author Shikhar Bhardwaj
 *
 * Constant step size policy for parallel Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_CONSTANT_STEP_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_CONSTANT_STEP_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Implementation of the ConstantStep stepsize decay policy for parallel SGD.
 */
class ConstantStep
{
 public:
  /**
   * Member initialization constructor.
   *
   * The defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.
   *
   * @param step The intial stepsize to use.
   */
  ConstantStep(const double step = 0.01) : step(step) { /* Nothing to do */ }

  /**
   * This function is called in each iteration before the gradient update.
   *
   * @param numEpoch The iteration number for which the stepsize is to be
   *    calculated.
   * @return The step size for the current iteration.
   */
  double StepSize(const size_t /* numEpoch */)
  {
    return step;
  }
 private:
  //! The initial stepsize, which remains unchanged
  double step;
};

} // namespace optimization
} // namespace mlpack

#endif
