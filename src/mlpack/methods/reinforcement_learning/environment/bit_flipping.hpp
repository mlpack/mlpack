/**
 * @file methods/reinforcement_learning/environment/bit_flipping.hpp
 * @author Eshaan Agarwal
 *
 * This file is an implementation of Bit Flipping toy task:
 * https://github.com/NervanaSystems/gym-bit-flip/blob/master/gym_bit_flip/bit_flip.py
 * https://github.com/ceteke/her/blob/master/env.py
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_BIT_FLIPPING_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_BIT_FLIPPING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of Cart Pole task.
 */
class BitFlipping
{
 public:
  /**
   * Implementation of the state of Bit Flipping. Each state is a 
   * (position, velocity, angle, angular velocity).
   */
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
     * @param data Data 
     */
    State(const arma::vec& data) : data(data)
    { /* Nothing to do here */ }

    //! Modify the internal representation of the state.
    arma::vec& Data() { return data; }

    //! Get the internal representation of the state.
    arma::vec Data() const { return data; }

    //! Encode the state to a column vector.
    const arma::vec& Encode() const { return data; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 10;

   private:
    //! Locally-stored n bit integer.
    arma::vec data;
  };

  /**
   * Implementation of action of Cart Pole.
   */
  class Action
  {
   public:
    
    // To store the action ( index of n bit number to be flipped )
    size_t action;

    // Track the size of the action space.
    static const size_t size = 1;
  };

  /**
   * Construct a binary vector instance using the given constants.
   *
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
   * @param length Length of the binary vector for state and goal
   */
  BitFlipping(const size_t maxSteps = 200,
              const size_t length = 10) :
      maxSteps(maxSteps),
      length(length),
      stepsPerformed(0)
  { 
    
  }

  /**
   * Get reward and next state based on current
   * state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState,
                const State& transitionGoal)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    // // Modify state according to action
    nextState.Data()= state.Data();
    nextState.Data()(action.action) = 1 - state.Data()(action.action);

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Do not reward agent if it failed.
    if (done && arma::approx_equal(state.Data(), goal, "absdiff", 1e-5))
      return 1.0;

    // Reward agent if it reaches the goal of transition and is not done
    if (!done && arma::approx_equal(nextState.Data(), transitionGoal.Data(), "absdiff", 1e-5))
    {
      return 1.0;
    }

    return 0.0;
  }

  /**
   * Get reward based on current state and current
   * action.
   *
   * @param state The current state.
   * @param action The current action.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state, const Action& action, const State& transitionGoal)
  {
    State nextState;
    return Sample(state, action, nextState, transitionGoal);
  }

  /**
   * Initial state representation is randomly generated binary vector
   *
   * @return Initial state for each episode.
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    initialState = arma::randi<arma::vec>(length, arma::distr_param(0, 1));
    return State(initialState);
  }

  /**
   * Get reward for particular goal
   *
   * @param nextState The next state.
   * @param transitionGoal Transition's goal.
   * @return Initial state for each episode.
   */
  double GetHERReward(const State& nextState,
                    const State& transitionGoal)
  {
    // Reward agent if it reaches the goal of transition and is not done
    if (arma::approx_equal(nextState.Data(), transitionGoal.Data(), "absdiff", 1e-5))
    {
      return 1.0;
    }

    return 0.0;
  }

  /**
   * This function checks if the cart has reached the terminal state.
   *
   * @param state The desired state.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const
  { 
    if (maxSteps != 0 && stepsPerformed >= maxSteps)
    {
      Log::Info << "Episode terminated due to the maximum number of steps"
          "being taken.";
      return true;
    }
    else if (arma::approx_equal(state.Data(), goal, "absdiff", 1e-5))
    {
      Log::Info << "Episode terminated as agent has reached desired goal.";
      return true;
    }
    return false;
  }

  /**
   * Initial goal representation for thr environment
   *
   * @return Initial goal for each episode.
   */
  State GoalSample()
  {
    // goal = arma::randi<arma::colvec>(length, arma::distr_param(0, 1));
    do
    {
      goal = arma::randi<arma::vec>(length, arma::distr_param(0, 1));
    }
    while (sum(initialState - goal) == 0);
    return State(goal);
  }

  //! Get the number of steps performed.
  size_t StepsPerformed() const { return stepsPerformed; }

  //! Get the maximum number of steps allowed.
  size_t MaxSteps() const { return maxSteps; }
  //! Set the maximum number of steps allowed.
  size_t& MaxSteps() { return maxSteps; }

  //! Get the size of binary vector.
  size_t Length() const { return length; }
  //! Set the size of binary vector
  size_t& Length() { return length; }

  //! Get the goal for the episode
  arma::vec Goal() const { return goal; }
  //! Set the goal for the episode
  arma::vec& Goal() { return goal; }

 private:
  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored done reward.
  double doneReward;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;

  //! Locally stored goal for the epsiode
  arma::vec goal;

  //! Locally stored initialState for the epsiode
  arma::vec initialState;

  //! Locally stored size of binary vector
  size_t length;
};

} // namespace mlpack

#endif
