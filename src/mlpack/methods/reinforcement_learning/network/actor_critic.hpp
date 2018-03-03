/**
 * @file actor_critic.hpp
 * @author Shangtong Zhang
 *
 * Definition and implementation of the ActorCritic network class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_NETWORK_ACTOR_CRITIC_HPP
#define MLPACK_METHODS_RL_NETWORK_ACTOR_CRITIC_HPP

#include <mlpack/core.hpp>

using namespace mlpack::ann;

namespace mlpack {
namespace rl {

/**
 * This is a wrapper of actor network and critic network. Note that this wrapper
 * doesn't support shared layers between the two networks.
 *
 * @tparam ActorType The type of the actor network.
 * @tparam CriticType The type of the critic network.
 */
template <
  typename ActorType,
  typename CriticType
>
class ActorCriticNetwork
{
 public:
  /**
   * Create an empty wrapper.
   */
  ActorCriticNetwork()
  { /* Nothing to do here. */ }

  /**
   * Create a wrapper network with given model.
   *
   * @param actor The actor network.
   * @param critic The critic network.
   */
  ActorCriticNetwork(ActorType actor, CriticType critic) :
      actor(actor), critic(critic)
  { /* Nothing to do here. */ }

  /**
   * @return Whether the network is initialized.
   */
  bool Initialized() { return actor.Initialized() && critic.Initialized(); }

  //! Get the actor network.
  const ActorType& Actor() const { return actor; }

  //! Modify the actor network.
  ActorType& Actor() { return actor; }

  //! Get the critic network.
  const CriticType& Critic() const { return critic; }

  //! Modify the critic network.
  CriticType& Critic() { return critic; }

  //! Reset the parameters of the network.
  void ResetParameters()
  {
    actor.ResetParameters();
    critic.ResetParameters();
  }

 private:
  //! Locally-stored actor network.
  ActorType actor;

  //! Locally-stored critic network.
  CriticType critic;
};

} // namespace rl
} // namespace mlpack

#endif
