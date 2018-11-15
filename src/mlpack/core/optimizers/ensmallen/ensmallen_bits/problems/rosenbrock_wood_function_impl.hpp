/**
 * @file rosenbrock_wood_function_impl.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Rosenbrock-Wood function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "rosenbrock_wood_function.hpp"

namespace ens {
namespace test {

inline RosenbrockWoodFunction::RosenbrockWoodFunction() : rf(4), wf()
{
  initialPoint.set_size(4, 2);
  initialPoint.col(0) = rf.GetInitialPoint();
  initialPoint.col(1) = wf.GetInitialPoint();
}

inline void RosenbrockWoodFunction::Shuffle() { /* Nothing to do here */ }

inline double RosenbrockWoodFunction::Evaluate(const arma::mat& coordinates,
                                               const size_t /* begin */,
                                               const size_t /* batchSize */)
    const
{
  const double objective = rf.Evaluate(coordinates.col(0)) +
      wf.Evaluate(coordinates.col(1));

  return objective;
}

inline double RosenbrockWoodFunction::Evaluate(const arma::mat& coordinates)
    const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void RosenbrockWoodFunction::Gradient(const arma::mat& coordinates,
                                             const size_t /* begin */,
                                             arma::mat& gradient,
                                             const size_t /* batchSize */) const
{
  gradient.set_size(4, 2);

  arma::vec grf(4);
  arma::vec gwf(4);

  rf.Gradient(coordinates.col(0), grf);
  wf.Gradient(coordinates.col(1), gwf);

  gradient.col(0) = grf;
  gradient.col(1) = gwf;
}

inline void RosenbrockWoodFunction::Gradient(const arma::mat& coordinates,
                                             arma::mat& gradient) const
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
