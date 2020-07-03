/**
 * @file methods/reinforcement_learning/environment/env_type.hpp
 * @author Nishant Kumar
 *
 * This file defines a dummy environment to be use with gym_tcp_api.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_ENV_TYPE_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_ENV_TYPE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

class DiscreteActionEnv
{
 public:
  class State
  {
   public:
    /**
     * Construct a state instance.
     */
    State() : data(dimension)
    { /* Nothing to do here. */ }

    /**
     * Construct a state instance from given data.
     *
     * @param data Data for the state.
     */
    State(const arma::colvec& data) : data(data)
    { /* Nothing to do here */ }

    //! Modify the internal representation of the state.
    arma::colvec& Data() { return data; }

    //! Encode the state to a column vector.
    const arma::colvec& Encode() const { return data; }

    //! Dimension of the encoded state.
    static size_t dimension;

   private:
    //! Locally-stored state data.
    arma::colvec data;
  };
  /**
   * Implementation of action of Cart Pole.
   */
  enum Action
  {
    backward,
    forward,

    // Track the size of the action space.
    size
  };
  double Sample(const State& state,
                const Action& action,
                State& nextState)
  {
    return 1.0;
  }

  State InitialSample()
  {
    return State();
  }

  /**
   * This function checks if the cart has reached the terminal state.
   *
   * @param state The desired state.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const
  {
    return false;
  }
};
size_t DiscreteActionEnv::State::dimension = 0;

} // namespace rl
} // namespace mlpack

#endif
