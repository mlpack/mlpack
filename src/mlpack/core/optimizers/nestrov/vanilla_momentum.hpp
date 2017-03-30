/**
 * @file nestrov_update.hpp
 * @author Kris Singh
 *
 * Nestrov Momentum update for Stochastic Gradient Descent.
 * 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_MOMENTUM_VANILLA_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_MOMENTUM_VANILLA_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {
class VanillaMomentum
{
public:
  VanillaMomentum()
  {/* Do Nothing */}

  void UpdateMomemntum(const double& momentum, double& result)
  {
    result = momentum;
  }

};
}// namespace optimisation
}// namespace mlpack
#endif
