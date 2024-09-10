/**
 * @file methods/reinforcement_learning/environment/ftn.hpp
 * @author Nanubala Gnana Sai
 *
 * This file is an implementation of Fruit Tree Navigation (FTN) Task:
 * https://github.com/RunzheYang/MORL/blob/master/synthetic/envs/fruit_tree.py
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_FTN_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_FTN_HPP

#include <mlpack/core.hpp>
#include "ftn_param.hpp"

namespace mlpack {

/**
 * Implementation of Fruit Tree Navigation Task.
 * For more details, see the following:
 * @code
 * @incollection{yang2019morl,
 *   title = {A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation},
 *   author = {Yang, Runzhe and Sun, Xingyuan and Narasimhan, Karthik},
 *   booktitle = {Advances in Neural Information Processing Systems 32},
 *   pages = {14610--14621},
 *   year = {2019}
 * }
 * @endcode
 */
class FruitTreeNavigation
{
 public:
  /**
   * Implementation of Fruit Tree Navigation state. Each State is a tuple 
   * {(row, column)} representing zero based index of a node in the tree.
   */
  class State
  {
   public:
    /**
     * Construct a state instance.
     */
    State(): data(dimension) { /* Nothing to do here. */ }

    /**
     * Construct a state instance from given data.
     *
     * @param data Data for the zero based index of the current node.
     */
    State(const arma::colvec& data) : data(data)
    { /* Nothing to do here. */ }

    //! Modify the state representation.
    arma::vec& Data() { return data; }

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
    //! Locally-Stored {(row, column)}.
    arma::vec data;
  };

  /**
   * Implementation of action for Fruit Tree Navigation task.
   */
  class Action
  {
   public:
    enum actions
    {
      left,
      right,
    };
    // To store the action.
    Action::actions action;

    // Track the size of the action space.
    static const size_t size = 2;
  };

  /**
   * Construct a Fruit Tree Navigation instance using the given constants.
   *
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
   * @param depth The maximum depth of the full binary tree.
   */
  FruitTreeNavigation(const size_t maxSteps = 500,
      const size_t depth = 6) :
      maxSteps(maxSteps),
      stepsPerformed(0),
      fruitTree(depth)
  {
    // Nothing to do.
  }

  /**
   * Dynamics of the FTN System. To get reward and next state based on
   * current state and current action. Return null vector reward as default.
   *
   * @param state The current State.
   * @param action The action taken.
   * @param nextState The next state.
   * @return reward, defaults to null vector.
   */
  arma::vec Sample(const State& state,
                   const Action& action,
                   State& nextState)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    // Make a vector to estimate nextstate.
    arma::vec currentState {state.Row(), state.Column()};
    arma::vec direction = std::unordered_map<Action::actions, arma::vec>({
            { Action::actions::left,  arma::vec({1, currentState(1)}) },
            { Action::actions::right,  arma::vec({1, currentState(1) + 1}) }
        })[action.action];

    arma::vec currentNextState = currentState + direction;
    nextState.Row() = currentNextState[0];
    nextState.Column() = currentNextState[1];

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Do not reward the agent if time ran out.
    if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
      return zeros(rewardSize);

    return fruitTree.GetReward(state);
  };

  /**
   * Dynamics of the FTN System. To get reward and next state based on
   * current state and current action. This function calls the Sample function
   * to estimate the next state return reward for taking a particular action.
   *
   * @param state The current State.
   * @param action The action taken.
   * @return nextState The next state.
   */
  arma::vec Sample(const State& state, const Action& action)
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  /**
   * This function does null initialization of state space.
   * Init to the root of the tree (0, 0).
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    return State(zeros<arma::vec>(2));
  }

  /**
   * This function checks if the FTN has reached the terminal state.
   *
   * @param state The current State.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const
  {
    if (maxSteps != 0 && stepsPerformed >= maxSteps)
    {
      Log::Info << "Episode terminated due to the maximum number of steps"
          " being taken.";
      return true;
    }
    else if (state.Row() == fruitTree.Depth())
    {
      Log::Info << "Episode terminated due to reaching leaf node.";
      return true;
    }
    return false;
  }

  //! Get the number of steps performed.
  size_t StepsPerformed() const { return stepsPerformed; }

  //! Get the maximum number of steps allowed.
  size_t MaxSteps() const { return maxSteps; }
  //! Set the maximum number of steps allowed.
  size_t& MaxSteps() { return maxSteps; }

  //! The reward vector consists of {Protein, Carbs, Fats, Vitamins, Minerals,
  //! Water}. A total of 6 rewards.
  static constexpr size_t rewardSize = 6;

 private:
  /**
   * A Fruit Tree is a full binary tree. Each node of this tree represents a
   * state.  Non-leaf nodes yield a null vector R^{6} reward. Leaf nodes have a
   * pre-defined set of reward vectors such that they lie on a Convex
   * Convergence Set (CCS).
   */
  class FruitTree
  {
   public:
    FruitTree(const size_t depth) : depth(depth)
    {
      if (std::find(validDepths.begin(), validDepths.end(), depth) ==
          validDepths.end())
      {
        throw std::logic_error("FruitTree()::FruitTree: Invalid depth value: " +
            std::to_string(depth) + " provided. Only depth values of: 5, 6, 7 "
            "are allowed.");
      }

      arma::mat branches = zeros(rewardSize, (size_t) std::pow(2, depth - 1));
      tree = join_rows(branches, Fruits());
    }

    // Extract array index from {(row, column)} representation.
    size_t GetIndex(const State& state) const
    {
      return static_cast<size_t>(std::pow(2, state.Row() - 1) + state.Column());
    }

    // Yield reward from the current node.
    arma::vec GetReward(const State& state) const
    {
      return tree.col(GetIndex(state));
    }

    //! Retreive the reward vectors of the leaf nodes.
    arma::mat Fruits() const { return ConvexSetMap.at(depth); }

    //! Retreive the maximum depth of the FTN tree.
    size_t Depth() const { return depth; }

   private:
    //! Maximum depth of the tree.
    size_t depth;

    //! Matrix representation of the internal tree.
    arma::mat tree;

    //! Valid depth values for the tree.
    const std::array<size_t, 3> validDepths {5, 6, 7};
  };

  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;

  //! Locally-stored fruit tree representation.
  FruitTree fruitTree;
};

} // namespace mlpack

#endif
