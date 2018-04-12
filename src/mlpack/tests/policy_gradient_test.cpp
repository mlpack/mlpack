/**
 * @file policygrad_test.hpp
 * @author Rohan Raj
 * 
 * Test for Policy Gradient Method implementation
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
#include <mlpack/methods/reinforcement_learning/policy_gradient.hpp>
#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/policy/stochiastic_policy.hpp>
#include <mlpack/core/optimizers/sga/sga_update.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(PolicyGradientTest);

//! Test Policy Gradient in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithPolicyGradient)
{
  // Set up the network.
  FFN<NegativeLogLikelihood<>, GaussianInitialization> model(NegativeLogLikelihood<>(),
      GaussianInitialization(0, 0.001));
  model.Add<Linear<>>(4, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 2);
  model.Add<LogSoftMax<> >();

  // Set up the policy and replay method.
  StochiasticPolicy<CartPole> policy;

  TrainingConfig config;
  config.StepSize() = 0.01;
  config.Discount() = 0.99;
  config.StepLimit() = 300;  

  // Set up Policy Gradient agent.
  PolicyGradient<CartPole, decltype(model), SgaUpdate, decltype(policy)>
      agent(std::move(config), std::move(model), std::move(policy));

  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;
  while (true)
  {
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes += 1;

    if (episodes > 1000)
    {
      Log::Debug << "Cart Pole with Policy Gradient method failed."
        << std::endl;
      converged = false;
      break;
    }

    /**
     * Reaching running average return 25 is sufficient to show it works.
     * For now I am just showing this will work .
     */
    Log::Debug << "Average return: " << averageReturn.mean()
        << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > 30)
    {
      agent.Deterministic() = true;
      arma::running_stat<double> testReturn;
      for (size_t i = 0; i < 10; ++i)
        testReturn(agent.Episode());

      Log::Debug << "Average return in deterministic test: "
          << testReturn.mean() << std::endl;
      break;
    }
  }
  // converged = true;
  BOOST_REQUIRE(converged);
}
BOOST_AUTO_TEST_SUITE_END();
