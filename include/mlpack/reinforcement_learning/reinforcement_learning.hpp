/**
 * @file methods/reinforcement_learning/reinforcement_learning.hpp
 * @author Ryan Curtin
 *
 * Convenience include for all mlpack's reinforcement learning code.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_REINFORCEMENT_LEARNING_REINFORCEMENT_LEARNING_HPP
#define MLPACK_METHODS_REINFORCEMENT_LEARNING_REINFORCEMENT_LEARNING_HPP

#include "environment/environment.hpp"
#include "policy/policy.hpp"
#include "q_networks/q_networks.hpp"
#include "replay/replay.hpp"
#include "worker/worker.hpp"
#include "noise/noise.hpp"

#include "training_config.hpp"
#include "async_learning.hpp"
#include "q_learning.hpp"
#include "ddpg.hpp"
#include "td3.hpp"
#include "sac.hpp"

#endif
