/**
 * @file methods/reinforcement_learning/environment/maze.hpp
 * @author Eshaan Agarwal
 *
 * This file is an implementation of Goal based Maze task:
 * 
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_MAZE_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_MAZE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of Maze task.
 */
class Maze
{
 public:
  /**
   * Implementation of the state of Maze.
   * Each State is a tuple {(row, column)} in n * n maze matrix
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

    //! Get value of row index of the node.
    double Row() const { return data[0]; }
    //! Modify value of row index of the node.
    double& Row() { return data[0]; }

    //! Get value of column index of the node.
    double Column() const { return data[1]; }
    //! Modify value of column index of the node.
    double& Column() { return data[1]; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 2;

   private:
    //! Locally-stored {(row, column)}.
    arma::vec data;
  };

  /**
   * Implementation of action of Maze.
   */
  class Action
  {
   public:
   enum actions
    {
      left,
      right,
      down,
      up,
    };
    
    // To store the action ( direction to move )
    Action::actions action;

    // Track the size of the action space.
    static const size_t size = 4;
  };

  /**
   * Construct a maze instance using the given constants.
   *
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
   */
  Maze(const size_t maxSteps = 100) :
      maxSteps(maxSteps),
      stepsPerformed(0)
  { 
    maze = { { 0, 0, 0 ,-1 },
             { -1, -1, 0, -1 },
             { 0, 0, 0, 0 }, 
             { 1, 0, 0, -1 }};
    
    directions = std::unordered_map<Action::actions, arma::vec>({
            { Action::actions::left,  arma::vec({0, -1}) },
            { Action::actions::right,  arma::vec({0, 1}) },
            { Action::actions::up,  arma::vec({-1, 0}) },
            { Action::actions::down,  arma::vec({1, 0}) }
        });

  }

  /**
   * Get reward and next state based on current
   * state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @param transitionGoal The goal for transition
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState,
                const State& transitionGoal)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    // Make a vector to estimate nextstate.
    arma::vec currentState {state.Row(), state.Column()};
    arma::vec direction = directions[action.action];

    arma::vec currentNextState = currentState + direction;
    nextState.Row() = currentNextState[0];
    nextState.Column() = currentNextState[1];

    // dont move to that position if invalid
    bool invalid = false;

    if(currentNextState(0)>= maze.n_rows || currentNextState(1)>= maze.n_cols
        || currentNextState(0) < 0 || currentNextState(1) < 0 || 
        maze(currentNextState(0),currentNextState(1))==-1 ){
        invalid = true;
        nextState.Row() = currentState[0];
        nextState.Column() = currentState[1];
    }   

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Do not reward agent if it failed.
    if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
      return 0.0;
    
    if (done && maze(nextState.Row(),nextState.Column()) == 1 && 
        nextState.Row() == goal(0) && nextState.Column() == goal(1) )
      return 1.0;

    return 0.0;
  }

  /**
   * Get reward based on current state and current
   * action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param transitionGoal The transition goal.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state, const Action& action,
                 const State& transitionGoal)
  {
    State nextState;
    return Sample(state, action, nextState, transitionGoal);
  }

  /**
   * Initial state (row,column) for agent in maze
   *
   * @return Initial state for each episode.
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    startingPoints = {arma::vec({0,0}), arma::vec({0,1}), arma::vec({0,2}),
                       arma::vec({1,2}), arma::vec({2,0}), arma::vec({2,1}),
                        arma::vec({2,2}), arma::vec({2,3}), arma::vec({3,1}),
                         arma::vec({3,2})};
    size_t index = arma::randi(arma::distr_param(0, startingPoints.size() -1));
    initialState = arma::vec({startingPoints[index][0],startingPoints[index][1]});
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
    if (maze(nextState.Row(),nextState.Column()) == 0 && 
         nextState.Row() == transitionGoal.Row() && 
         nextState.Column() == transitionGoal.Column() )
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
    else if (maze(state.Row(),state.Column()) == 1)
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
    goal = arma::vec({3,0});
    return State(goal);
  }

  //! Get the number of steps performed.
  size_t StepsPerformed() const { return stepsPerformed; }

  //! Get the maximum number of steps allowed.
  size_t MaxSteps() const { return maxSteps; }
  //! Set the maximum number of steps allowed.
  size_t& MaxSteps() { return maxSteps; }

  //! Get the goal for the episode
  arma::vec Goal() const { return goal; }
  //! Set the goal for the episode
  arma::vec& Goal() { return goal; }

 private:
  //! 4*4 Maze generated 
  arma::mat maze;

  //! Starting points from where agent can start
  std::vector<arma::vec> startingPoints;

  //! allowed directions and associated moves
  std::unordered_map<Action::actions, arma::vec> directions;

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

};

} // namespace mlpack

#endif
