
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_replay_prioritized_replay.hpp:

Program Listing for File prioritized_replay.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_replay_prioritized_replay.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/replay/prioritized_replay.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_PRIORITIZED_REPLAY_HPP
   #define MLPACK_METHODS_RL_PRIORITIZED_REPLAY_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "sumtree.hpp"
   
   namespace mlpack {
   namespace rl {
   
   template <typename EnvironmentType>
   class PrioritizedReplay
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
   
     PrioritizedReplay():
         batchSize(0),
         capacity(0),
         position(0),
         full(false),
         alpha(0),
         maxPriority(0),
         initialBeta(0),
         beta(0),
         replayBetaIters(0),
         nSteps(0)
     { /* Nothing to do here. */ }
   
     PrioritizedReplay(const size_t batchSize,
                       const size_t capacity,
                       const double alpha,
                       const size_t nSteps = 1,
                       const size_t dimension = StateType::dimension) :
         batchSize(batchSize),
         capacity(capacity),
         position(0),
         full(false),
         alpha(alpha),
         maxPriority(1.0),
         initialBeta(0.6),
         replayBetaIters(10000),
         nSteps(nSteps),
         states(dimension, capacity),
         actions(capacity),
         rewards(capacity),
         nextStates(dimension, capacity),
         isTerminal(capacity)
     {
       size_t size = 1;
       while (size < capacity)
       {
         size *= 2;
       }
   
       beta = initialBeta;
       idxSum = SumTree<double>(size);
     }
   
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
   
       idxSum.Set(position, maxPriority * alpha);
   
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
   
     arma::ucolvec SampleProportional()
     {
       arma::ucolvec idxes(batchSize);
       double totalSum = idxSum.Sum(0, (full ? capacity : position));
       double sumPerRange = totalSum / batchSize;
       for (size_t bt = 0; bt < batchSize; bt++)
       {
         const double mass = arma::randu() * sumPerRange + bt * sumPerRange;
         idxes(bt) = idxSum.FindPrefixSum(mass);
       }
       return idxes;
     }
   
     void Sample(arma::mat& sampledStates,
                 std::vector<ActionType>& sampledActions,
                 arma::rowvec& sampledRewards,
                 arma::mat& sampledNextStates,
                 arma::irowvec& isTerminal)
     {
       sampledIndices = SampleProportional();
       BetaAnneal();
   
       sampledStates = states.cols(sampledIndices);
       for (size_t t = 0; t < sampledIndices.n_rows; t ++)
         sampledActions.push_back(actions[sampledIndices[t]]);
       sampledRewards = rewards.elem(sampledIndices).t();
       sampledNextStates = nextStates.cols(sampledIndices);
       isTerminal = this->isTerminal.elem(sampledIndices).t();
   
       // Calculate the weights of sampled transitions.
   
       size_t numSample = full ? capacity : position;
       weights = arma::rowvec(sampledIndices.n_rows);
   
       for (size_t i = 0; i < sampledIndices.n_rows; ++i)
       {
         double p_sample = idxSum.Get(sampledIndices(i)) / idxSum.Sum();
         weights(i) = pow(numSample * p_sample, -beta);
       }
       weights /= weights.max();
     }
   
     void UpdatePriorities(arma::ucolvec& indices, arma::colvec& priorities)
     {
         arma::colvec alphaPri = alpha * priorities;
         maxPriority = std::max(maxPriority, arma::max(priorities));
         idxSum.BatchUpdate(indices, alphaPri);
     }
   
     const size_t& Size()
     {
       return full ? capacity : position;
     }
   
     void BetaAnneal()
     {
       beta = beta + (1 - initialBeta) * 1.0 / replayBetaIters;
     }
   
     void Update(arma::mat target,
                 std::vector<ActionType> sampledActions,
                 arma::mat nextActionValues,
                 arma::mat& gradients)
     {
       arma::colvec tdError(target.n_cols);
       for (size_t i = 0; i < target.n_cols; i ++)
       {
         tdError(i) = nextActionValues(sampledActions[i].action, i) -
             target(sampledActions[i].action, i);
       }
       tdError = arma::abs(tdError);
       UpdatePriorities(sampledIndices, tdError);
   
       // Update the gradient
       gradients = arma::mean(weights) * gradients;
     }
   
     const size_t& NSteps() const { return nSteps; }
   
    private:
     size_t batchSize;
   
     size_t capacity;
   
     size_t position;
   
     bool full;
   
     double alpha;
   
     double maxPriority;
   
     double initialBeta;
   
     double beta;
   
     size_t replayBetaIters;
   
     SumTree<double> idxSum;
   
     arma::ucolvec sampledIndices;
   
     arma::rowvec weights;
   
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
