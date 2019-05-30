/**
 * @file ppo_test.hpp
 * @author Xiaohong Ji
 *
 * Test for PPO implementation
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/surrogate_loss.hpp>
#include <mlpack/methods/reinforcement_learning/environment/pendulum.hpp>

#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(PPOTEST);

//! Test PPO in Pendulum task.
BOOST_AUTO_TEST_CASE(PENDULUMWITHPPO)
{
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> critic(MeanSquaredError<>(),
      GaussianInitialization(0, 0.001));

  critic.Add<Linear<>>(4, 128);
  critic.Add<ReLULayer<>>();
  critic.Add<Linear<>>(128, 1);

  FFN<SurrogateLoss<>, GaussianInitialization> actor(SurrogateLoss<>(),
                                                         GaussianInitialization(0, 0.001));
  actor.Add<Linear<>>(4, 128);
  actor.Add<ReLULayer<>>();
  actor.Add<Linear<>>(128, 2);
}

BOOST_AUTO_TEST_SUITE_END();
