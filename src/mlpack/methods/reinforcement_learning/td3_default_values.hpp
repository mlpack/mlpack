/**
 * @file methods/reinforcement_learning/td3_default_values.hpp
 * @author Advaith P
 *
 * This file has default values for the TD3 class,which implements the 
 * Twin Delayed Deep Deterministic Policy Gradient algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef DEFAULT_VALUES_RL_TD3
#define DEFAULT_VALUES_RL_TD3



#include <mlpack/core.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include "training_config.hpp"

using namespace mlpack;
using namespace ens;


namespace DefaultValues 
{

inline TrainingConfig TrainingConfigWithDefaults() {
    TrainingConfig config;
    config.StepSize() = 0.01;
    config.TargetNetworkSyncInterval() = 1;
    config.UpdateInterval() = 3;
    config.Rho() = 0.001;
    return config;
}

inline FFN<EmptyLoss, GaussianInitialization> qNetwork() {
    // Construct qNetwork instance
    FFN<EmptyLoss, GaussianInitialization> qNetworkInstance(EmptyLoss(), GaussianInitialization(0, 0.1));
    // Add layers
    qNetworkInstance.Add(new Linear(128));
    qNetworkInstance.Add(new ReLU());
    qNetworkInstance.Add(new Linear(1));
    return qNetworkInstance;
}

inline FFN<EmptyLoss, GaussianInitialization> policyNetwork() {
    // Construct policyNetwork instance
    FFN<EmptyLoss, GaussianInitialization> policyNetworkInstance(EmptyLoss(), GaussianInitialization(0, 0.1));
    // Add layers
    policyNetworkInstance.Add(new Linear(128));
    policyNetworkInstance.Add(new ReLU());
    policyNetworkInstance.Add(new Linear(1));
    policyNetworkInstance.Add(new TanH());
    return policyNetworkInstance;
}


AdamUpdate default_updater;




}
#endif


