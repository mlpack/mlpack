/**
 * @file ppo.hpp
 * @author Xiaohong Ji
 *
 * This file is the implementation of PPO class, which implements
 * proximal policy optimization algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_PPO_IMPL_HPP
#define MLPACK_METHODS_RL_PPO_IMPL_HPP

#include <mlpack/prereqs.hpp>

#include "ppo.hpp"

namespace mlpack {
namespace rl {

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType = RandomReplay<EnvironmentType>
>
PPO<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::PPO(TrainingConfig config,
       NetworkType network,
       PolicyType policy,
       ReplayType replayMethod,
       UpdaterType updater,
       EnvironmentType environment):
{

}

#endif
