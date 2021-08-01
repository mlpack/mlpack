
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_replay_random_replay.hpp:

Program Listing for File random_replay.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_replay_random_replay.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/replay/random_replay.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_REPLAY_RANDOM_REPLAY_HPP
   #define MLPACK_METHODS_RL_REPLAY_RANDOM_REPLAY_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <cassert>
   
   namespace mlpack {
   namespace rl {
   
   template <typename EnvironmentType>
   class RandomReplay
   {
    public:
     using ActionType = typename EnvironmentType::Action;
   
     using StateType = typename EnvironmentType::State;
   
     struct Transition
     {
       StateType state;
       ActionType action;
       double reward;
       StateType nextState;
       bool isEnd;
     };
   
     RandomReplay():
         batchSize(0),
         capacity(0),
         position(0),
         full(false),
         nSteps(0)
     { /* Nothing to do here. */ }
   
     RandomReplay(const size_t batchSize,
                  const size_t capacity,
                  const size_t nSteps = 1,
                  const size_t dimension = StateType::dimension) :
         batchSize(batchSize),
         capacity(capacity),
         position(0),
         full(false),
         nSteps(nSteps),
         states(dimension, capacity),
         actions(capacity),
         rewards(capacity),
         nextStates(dimension, capacity),
         isTerminal(capacity)
     { /* Nothing to do here. */ }
   
     void Store(StateType state,
                ActionType action,
                double reward,
                StateType nextState,
                bool isEnd,
                const double& discount)
     {
       nStepBuffer.push_back({state, action, reward, nextState, isEnd});
   
       // Single step transition is not ready.
       if (nStepBuffer.size() < nSteps)
         return;
   
       // To keep the queue size fixed to nSteps.
       if (nStepBuffer.size() > nSteps)
         nStepBuffer.pop_front();
   
       // Before moving ahead, lets confirm if our fixed size buffer works.
       assert(nStepBuffer.size() == nSteps);
   
       // Make a n-step transition.
       GetNStepInfo(reward, nextState, isEnd, discount);
   
       state = nStepBuffer.front().state;
       action = nStepBuffer.front().action;
   
       states.col(position) = state.Encode();
       actions[position] = action;
       rewards(position) = reward;
       nextStates.col(position) = nextState.Encode();
       isTerminal(position) = isEnd;
       position++;
       if (position == capacity)
       {
         full = true;
         position = 0;
       }
     }
   
     void GetNStepInfo(double& reward,
                       StateType& nextState,
                       bool& isEnd,
                       const double& discount)
     {
       reward = nStepBuffer.back().reward;
       nextState = nStepBuffer.back().nextState;
       isEnd = nStepBuffer.back().isEnd;
   
       // Should start from the second last transition in buffer.
       for (int i = nStepBuffer.size() - 2; i >= 0; i--)
       {
         bool iE = nStepBuffer[i].isEnd;
         reward = nStepBuffer[i].reward + discount * reward * (1 - iE);
         if (iE)
         {
           nextState = nStepBuffer[i].nextState;
           isEnd = iE;
         }
       }
     }
   
     void Sample(arma::mat& sampledStates,
                 std::vector<ActionType>& sampledActions,
                 arma::rowvec& sampledRewards,
                 arma::mat& sampledNextStates,
                 arma::irowvec& isTerminal)
     {
       size_t upperBound = full ? capacity : position;
       arma::uvec sampledIndices = arma::randi<arma::uvec>(
           batchSize, arma::distr_param(0, upperBound - 1));
   
       sampledStates = states.cols(sampledIndices);
       for (size_t t = 0; t < sampledIndices.n_rows; t ++)
         sampledActions.push_back(actions[sampledIndices[t]]);
       sampledRewards = rewards.elem(sampledIndices).t();
       sampledNextStates = nextStates.cols(sampledIndices);
       isTerminal = this->isTerminal.elem(sampledIndices).t();
     }
   
     const size_t& Size()
     {
       return full ? capacity : position;
     }
   
     void Update(arma::mat /* target */,
                 std::vector<ActionType> /* sampledActions */,
                 arma::mat /* nextActionValues */,
                 arma::mat& /* gradients */)
     {
       /* Do nothing for random replay. */
     }
   
     const size_t& NSteps() const { return nSteps; }
   
    private:
     size_t batchSize;
   
     size_t capacity;
   
     size_t position;
   
     bool full;
   
     size_t nSteps;
   
     std::deque<Transition> nStepBuffer;
   
     arma::mat states;
   
     std::vector<ActionType> actions;
   
     arma::rowvec rewards;
   
     arma::mat nextStates;
   
     arma::irowvec isTerminal;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
