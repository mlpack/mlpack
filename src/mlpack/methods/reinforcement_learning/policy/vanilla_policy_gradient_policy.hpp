/**
 * @file vanilla policy gradient.hpp
 * @author Rohan Raj
 *
 * This file is an implementation of vanilla policy gradient.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_VANILLA_POLICY_GRADIENT_HPP
#define MLPACK_METHODS_RL_VANILLA_POLICY_GRADIENT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation for vanilla policy gradient.
 *
 * In general, we are developing an random float number in 0 to 1 and
 * checking if it is less than our softmax output. If yes then , then 
 * play randomly . 
 *
 * @tparam EnvironmentType The reinforcement learning task.
 */
template <typename EnvironmentType>
class VanillaPolicyGradient
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  /**
   * Constructor for epsilon greedy policy class.
   *
   * @param initialEpsilon The initial probability to explore
   *        (select a random action).
   * @param annealInterval The steps during which the probability to explore
   *        will anneal.
   * @param minEpsilon Epsilon will never be less than this value.
   */
  VanillaPolicyGradient(const double initialEpsilon =math::Random(),
               const size_t annealInterval = 100,
               const double minEpsilon = math::Random()) :
      epsilon(initialEpsilon),
      minEpsilon(minEpsilon),
      delta((initialEpsilon - minEpsilon) / annealInterval)
  { /* Nothing to do here. */ }

  /**
   * Sample an action based on given action values.
   *
   * @param actionValue Values for each action.
   * @param deterministic Always select the action greedily.
   * @return Sampled action.
   */
  ActionType Sample(const arma::colvec& actionValue, bool deterministic = false)
  {
    double exploration = math::Random();

    // Select the action randomly.
    if (!deterministic && exploration > actionValue.max())
      return static_cast<ActionType>(math::RandInt(ActionType::size));

    // Select the action greedily.
    return static_cast<ActionType>(
        arma::as_scalar(arma::find(actionValue == actionValue.max(), 1)));
  }

  /**
   * Exploration probability will anneal at each step.
   */
  void Anneal()
  { /* Nothing to do here. */ };

  /**
   * @return Current possibility to explore.
   */
  const double& Epsilon() const { return epsilon; }

 private:
  //! Locally-stored probability to explore.
  double epsilon;

  //! Locally-stored lower bound for epsilon.
  double minEpsilon;

  //! Locally-stored stride for epsilon to anneal.
  double delta;
};

} // namespace rl
} // namespace mlpack

#endif
