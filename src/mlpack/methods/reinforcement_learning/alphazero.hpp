/**
 * @file methods/reinforcement_learning/alphazero.hpp
 * @author Antoine Dubois
 *
 * This file is the definition of AlphaZero class, which implements the
 * AlphaZero algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ALPHALIKE_HPP
#define MLPACK_METHODS_RL_ALPHALIKE_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "training_config.hpp"

namespace mlpack {

class AlphaZeroConfig : public TrainingConfig 
{
  public:
    AlphaZeroConfig() :
      TrainingConfig(),
      selfplaySteps(0),
      explorationUcb(1.41421356237) // square-root of two
    {
      // nothing to do here
    }

    AlphaZeroConfig(size_t numWorkers,
                    size_t updateInterval,
                    size_t stepLimit,
                    size_t explorationSteps, 
                    size_t selfplaySteps,
                    double stepSize, 
                    double vMin, 
                    double vMax,
                    double explorationUcb) :
      TrainingConfig(numWorkers, 
                     updateInterval, 
                     100, 
                     stepLimit,
                     explorationSteps,
                     stepSize,
                     0.99,
                     40,
                     false,
                     false,
                     false,
                     51,
                     vMin,
                     vMax,
                     0.005),
      selfplaySteps(selfplaySteps), 
      explorationUcb(explorationUcb) 
    {
      // nothing to do here
    }

    //! Get the number of selfplay steps.
    size_t SelfplaySteps() const { return selfplaySteps; }
    //! Modify the number of selfplay steps.
    size_t &SelfplaySteps() { return selfplaySteps; }

    //! Get the rho value for sac.
    double ExplorationUCB() const { return explorationUcb; }
    //! Modify the rho value for sac.
    double &ExplorationUCB() { return explorationUcb; }

  private:
    /**
     * Locally-stored selfplay steps before playing an episode step.
     * The agent udates the values and visit counts of state-action nodes
     * during these steps. Then, it plays a 'real' game’s step and update
     * the value and policy networks.
     * This is valid only for AlphaLike agent.
     */
    size_t selfplaySteps;
    /**
     * Locally-stored parameter for computing the UCB score.
     * The default value is 1.41421356237 the approximation of the square root of
     * 2 This is valid only for AlphaLike.
     */
    double explorationUcb;
};

/**
 * Implementation of AlphaZero.
 *
 * For more details, see the following:
 * @code
 * @misc{Lillicrap et al, 2015,
 *  author    = {Surag Nair},
 *  title     = {A Simple Alpha(Go) Zero Tutorial},
 *  year      = {2017},
 *  url       = {https://suragnair.github.io/posts/alphazero.html}
 * }
 * @endcode
 * The present implementation of AlphaZero is inspired by Ciamic’s
 * implementation at https://github.com/ciamic/alphazero/tree/main
 * @tparam EnvironmentType The environment of the reinforcement learning task.
 * @tparam PalueNetworkType The network used to estimate the state values.
 * @tparam PolicyNetworkType The network to compute action value.
 * @tparam UpdaterType How to apply gradients when training.
 * @tparam ReplayType Experience replay method.
 */
template <
  typename EnvironmentType,
  typename ValueNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType = RandomReplay<EnvironmentType, true>
>
class AlphaZero 
{
  public:
    //! Convenient typedef for state.
    using StateType = typename EnvironmentType::State;

    //! Convenient typedef for action.
    using ActionType = typename EnvironmentType::Action;

    /**
     * Create the AlphaZero object with given settings.
     *
     * If you want to pass in a parameter and discard the original parameter
     * object, you can directly pass the parameter, as the constructor takes
     * a reference. This avoids unnecessary copy.
     *
     * @param config Hyper-parameters for training.
     * @param learningValueNetwork The network to compute the state value.
     * @param policyNetwork The network to produce an action given a state.
     * @param replayMethod Experience replay method.
     * @param valueNetworkUpdater How to apply gradients to the value network when
     * training.
     * @param policyNetworkUpdater How to apply gradients to policy network
     *        when training.
     * @param environment Reinforcement learning task.
     */
    AlphaZero(AlphaZeroConfig &config, ValueNetworkType &valueNetwork,
              PolicyNetworkType &policyNetwork, ReplayType &replayMethod,
              EnvironmentType environment,
              UpdaterType valueNetworkUpdater = UpdaterType(),
              UpdaterType policyNetworkUpdater = UpdaterType());

    /**
     * Clean memory.
     */
    ~AlphaZero();

    /**
     * Update state values and visit counts through simulated state-action paths
     **/
    void SelfPlay();

    /**
     * Update the value and policy networks.
     * */
    void Update();

    /**
     * Select an action, given an agent.
     */
    void SelectAction();

    /**
     * Execute an episode.
     * @return Return of the episode.
     */
    double Episode();

    //! Modify total steps from beginning.
    size_t &TotalSteps() { return totalSteps; }
    //! Get total steps from beginning.
    const size_t &TotalSteps() const { return totalSteps; }

    //! Modify the state of the agent.
    StateType &State() { return state; }
    //! Get the state of the agent.
    const StateType &State() const { return state; }

    //! Get the action of the agent.
    const ActionType &Action() const { return action; }

  private:
    struct Node {
      Node **child;
      Node *parent;
      StateType state;
      double proba;
      double W;
      size_t N;
      bool isFullyExpanded;
      Node(StateType state, double proba = 1.0, Node *parent = nullptr)
          : state(state), proba(proba), parent(parent), child(nullptr), W(0.0),
            N(0), isFullyExpanded(false) {
        /* Defining the array of children as a vector
        for games with extremely large action spaces.
        */
      }
      ~Node() {
        if (child == nullptr)
          return;

        for (size_t a = 0; a < ActionType::size; ++a) {
          if (child[a] != nullptr) {
            delete child[a];
            child[a] = nullptr;
          }
        }
        delete[] child;
      }
      void DetachParent() {
        for (size_t a = 0; a < ActionType::size; ++a) {
          if ((parent->child[a] != nullptr) && (parent->child[a] != this)) {
            delete parent->child[a];
            parent->child[a] = nullptr;
          }
        }
        delete[] parent->child;
        parent = nullptr;
      }
      void AddChild(const size_t a, const StateType childState,
                    const double proba) {
        if (child == nullptr)
          child = new Node *[ActionType::size];
        child[a] = new Node(childState, proba, this);
      }
      void VerifyFullyExpanded() {
        isFullyExpanded = true;
        for (size_t i = 0; i < ActionType::size; ++i) {
          if (child[i] == nullptr) {
            isFullyExpanded = false;
            return;
          }
        }
      }
    };

    double PUCBscore(const Node *const node) {
      double q_value = node->W / double(node->N);
      double prior_score =
          node->proba * std::sqrt(double(node->parent->N)) / double(1 + node->N);
      return q_value + config.ExplorationUCB() * prior_score;
    }
    Node *BestPUCBchild(const Node *const parent) {
      double max_ucb = PUCBscore(parent->child[0]);
      size_t arg_max = 0;
      double score;
      for (size_t a = 1; a < ActionType::size; ++a) {
        score = PUCBscore(parent->child[a]);
        if (max_ucb < score) {
          max_ucb = score;
          arg_max = a;
        }
      }
      return parent->child[arg_max];
    }
    Node *Expand(Node *const parent) {
      // find the best child
      double maxLogProba = std::numeric_limits<double>::lowest();
      size_t argMax;
      arma::vec nnLogProba;
      policyNetwork.Predict(parent->state.Encode(), nnLogProba);

      for (size_t a = 0; a < ActionType::size; ++a) {
        // pass if the child has already been added to the node
        if ((parent->child != nullptr) && (parent->child[a] != nullptr))
          continue;

        if (nnLogProba[a] > maxLogProba) {
          maxLogProba = nnLogProba[a];
          argMax = a;
        }
      }

      // add the best child node to the tree
      StateType childState;
      ActionType childAction;
      childAction.action = static_cast<decltype(childAction.action)>(argMax);
      environment.Sample(parent->state, childAction, childState);
      parent->AddChild(argMax, childState, std::exp(maxLogProba));
      return parent->child[argMax];
    }
    double Rollout(const Node *const node) {
      arma::vec nn_v;
      valueNetwork.Predict(node->state.Encode(), nn_v);
      return arma::as_scalar(nn_v);
    }
    void Explore(Node *node) {
      /* Selection */
      while (node->isFullyExpanded) {
        node = BestPUCBchild(node);
      }
      /* Expansion */
      if (!environment.IsTerminal(node->state)) {
        node = Expand(node);
      }
      /* Rollout */
      double v = Rollout(node);
      /* Backpropagation */
      node->N++;
      node->W += v;
      node = node->parent;
      /* check if the last node of the selection phase
         is now fully expanded */
      node->VerifyFullyExpanded();
      while (node != nullptr) {
        node->N++;
        node->W += v;
        node = node->parent;
      }
    }
    void Next(Node *&node, ActionType &nextAction, arma::vec &probaNextAction) {
      size_t sum = 0;
      for (size_t i = 0, N; i < ActionType::size; ++i) {
        N = node->child[i]->N;
        sum += N;
        probaAction[i] = double(N);
      }
      double inv_sum = 1.0 / double(sum);
      for (size_t i = 0; i < ActionType::size; ++i) {
        probaAction[i] *= inv_sum;
      }

      size_t index = RandCategorical(probaAction);
      nextAction.action = static_cast<decltype(nextAction.action)>(index);
      node = node->child[index];
      node->DetachParent();
    }
    //! Min-Max rescalling of the sampled total returns
    void ScaleReturn(arma::mat &sampledRewards);
    //! Min-Max rescalling of the sampled total return
    void ScaleReturn(double &reward);

    //! Locally-stored hyper-parameters.
    AlphaZeroConfig &config;

    //! Locally-stored learning value network.
    ValueNetworkType &valueNetwork;

    //! Locally-stored policy network.
    PolicyNetworkType &policyNetwork;

    //! Locally-stored value network loss
    MeanSquaredError lossValueNetwork;

    //! Locally-stored policy network loss
    CrossEntropyLoss lossPolicyNetwork;

    //! Locally-stored experience method.
    ReplayType &replayMethod;

    //! Locally-stored updater.
    UpdaterType valueNetworkUpdater;
    #if ENS_VERSION_MAJOR >= 2
    typename UpdaterType::template Policy<arma::mat, arma::mat>
        *valueNetworkUpdatePolicy;
    #endif

    //! Locally-stored updater.
    UpdaterType policyNetworkUpdater;
    #if ENS_VERSION_MAJOR >= 2
    typename UpdaterType::template Policy<arma::mat, arma::mat>
        *policyNetworkUpdatePolicy;
    #endif

    //! Locally-stored reinforcement learning task.
    EnvironmentType environment;

    //! Locally-stored the rescalling factors.
    double r1;
    double r2;

    //! Total steps from the beginning of the task.
    size_t totalSteps;

    //! Locally-stored current state of the agent.
    StateType state;

    //! Locally-stored action of the agent.
    ActionType action;

    //! Locally-stored the probabilities to have selected the above action
    arma::vec probaAction;

    //! Locally-stored flag indicating training mode or test mode.
    bool deterministic;

    //! Locally-stored state-action node
    Node *node;
};

} // namespace mlpack

// Include implementation
#include "alphazero_impl.hpp"

#endif

