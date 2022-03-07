/**
 * @file methods/reinforcement_learning/environment/env_type.cpp
 * @author Nishant Kumar
 *
 * This file defines the static variables used by the discrete and continuous
 * environments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "env_type.hpp"

namespace mlpack {
namespace rl {

// Instantiate static members.

size_t DiscreteActionEnv::State::dimension = 0;
size_t DiscreteActionEnv::Action::size = 0;
size_t DiscreteActionEnv::rewardSize = 0;

size_t ContinuousActionEnv::State::dimension = 0;
size_t ContinuousActionEnv::Action::size = 0;
size_t ContinuousActionEnv::rewardSize = 0;

} // namespace rl
} // namespace mlpack
