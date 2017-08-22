#ifndef MLPACK_METHODS_RL_NETWORK_ACTOR_CRITIC_HPP
#define MLPACK_METHODS_RL_NETWORK_ACTOR_CRITIC_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/visitor/delete_visitor.hpp>

using namespace mlpack::ann;

namespace mlpack {
namespace rl {

template <
  typename ActorType,
  typename CriticType
>
class ActorCriticNetwork
{
public:
  ActorCriticNetwork() {}
  ActorCriticNetwork(ActorType actor, CriticType critic) : actor(actor), critic(critic) {}

  const arma::mat& Parameters() const { return actor.Parameters(); }
  arma::mat& Parameters() { return actor.Parameters(); }

  const ActorType& Actor() const { return actor; }
  ActorType& Actor() { return actor; }

  const CriticType& Critic() const { return critic; }
  CriticType& Critic() { return critic; }

  void ResetParameters()
  {
    actor.ResetParameters();
    critic.ResetParameters();
  }

private:
  ActorType actor;
  CriticType critic;
};

}
}

#endif
