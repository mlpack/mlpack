/**
 * @file methods/reinforcement_learning/td3.hpp
 * @author Ali Hossam
 *
 * This file contains a class called RLDefaultParams. This class holds 
 * default settings for different reinforcement learning algorithms. 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_DEFAULT_PARAMS
#define MLPACK_METHODS_RL_DEFAULT_PARAMS

#include <mlpack/core.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "training_config.hpp"
using namespace mlpack;
using namespace ens;

namespace mlpack
{

class RLdefaultParams
{
public:
  RLdefaultParams() {
    // Set up Training configuration.
    config.StepSize() = 0.01;
    config.TargetNetworkSyncInterval() = 1;
    config.UpdateInterval() = 3;
    config.Rho() = 0.001;

    // Set up Actor network.
    policyNetwork = FFN<EmptyLoss, GaussianInitialization>(
        EmptyLoss(), GaussianInitialization(0, 0.1));
    policyNetwork.Add(new Linear(128));
    policyNetwork.Add(new ReLU());
    policyNetwork.Add(new Linear(1));
    policyNetwork.Add(new TanH());

    // Set up Critic network.
    qNetwork = FFN<EmptyLoss, GaussianInitialization>(
        EmptyLoss(), GaussianInitialization(0, 0.1));
    qNetwork.Add(new Linear(128));
    qNetwork.Add(new ReLU());
    qNetwork.Add(new Linear(1));
  }

  // Training configuration.
  TrainingConfig config;

  // Actor and Critic networks.
  FFN<EmptyLoss, GaussianInitialization> policyNetwork;
  FFN<EmptyLoss, GaussianInitialization> qNetwork;
  
  // Updater
  AdamUpdate updater;
};

} // namespace mlpack

#endif
