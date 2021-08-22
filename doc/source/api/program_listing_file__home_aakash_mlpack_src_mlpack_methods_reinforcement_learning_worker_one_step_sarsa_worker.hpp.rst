
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_worker_one_step_sarsa_worker.hpp:

Program Listing for File one_step_sarsa_worker.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_worker_one_step_sarsa_worker.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/worker/one_step_sarsa_worker.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_WORKER_ONE_STEP_SARSA_WORKER_HPP
   #define MLPACK_METHODS_RL_WORKER_ONE_STEP_SARSA_WORKER_HPP
   
   #include <ensmallen.hpp>
   #include <mlpack/methods/reinforcement_learning/training_config.hpp>
   
   namespace mlpack {
   namespace rl {
   
   template <
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   class OneStepSarsaWorker
   {
    public:
     using StateType = typename EnvironmentType::State;
     using ActionType = typename EnvironmentType::Action;
     using TransitionType = std::tuple<StateType, ActionType, double, StateType,
         ActionType>;
   
     OneStepSarsaWorker(
         const UpdaterType& updater,
         const EnvironmentType& environment,
         const TrainingConfig& config,
         bool deterministic):
         updater(updater),
         #if ENS_VERSION_MAJOR >= 2
         updatePolicy(NULL),
         #endif
         environment(environment),
         config(config),
         deterministic(deterministic),
         pending(config.UpdateInterval())
     { Reset(); }
   
     OneStepSarsaWorker(const OneStepSarsaWorker& other) :
         updater(other.updater),
         #if ENS_VERSION_MAJOR >= 2
         updatePolicy(NULL),
         #endif
         environment(other.environment),
         config(other.config),
         deterministic(other.deterministic),
         steps(other.steps),
         episodeReturn(other.episodeReturn),
         pending(other.pending),
         pendingIndex(other.pendingIndex),
         network(other.network),
         state(other.state),
         action(other.action)
     {
       Reset();
   
       #if ENS_VERSION_MAJOR >= 2
       updatePolicy = new typename UpdaterType::template
           Policy<arma::mat, arma::mat>(updater,
                                        network.Parameters().n_rows,
                                        network.Parameters().n_cols);
       #endif
     }
   
     OneStepSarsaWorker(OneStepSarsaWorker&& other) :
         updater(std::move(other.updater)),
         #if ENS_VERSION_MAJOR >= 2
         updatePolicy(NULL),
         #endif
         environment(std::move(other.environment)),
         config(std::move(other.config)),
         deterministic(std::move(other.deterministic)),
         steps(std::move(other.steps)),
         episodeReturn(std::move(other.episodeReturn)),
         pending(std::move(other.pending)),
         pendingIndex(std::move(other.pendingIndex)),
         network(std::move(other.network)),
         state(std::move(other.state)),
         action(std::move(other.action))
     {
       #if ENS_VERSION_MAJOR >= 2
       other.updatePolicy = NULL;
   
       updatePolicy = new typename UpdaterType::template
           Policy<arma::mat, arma::mat>(updater,
                                        network.Parameters().n_rows,
                                        network.Parameters().n_cols);
       #endif
     }
   
     OneStepSarsaWorker& operator=(const OneStepSarsaWorker& other)
     {
       if (&other == this)
         return *this;
   
       #if ENS_VERSION_MAJOR >= 2
       delete updatePolicy;
       #endif
   
       updater = other.updater;
       environment = other.environment;
       config = other.config;
       deterministic = other.deterministic;
       steps = other.steps;
       episodeReturn = other.episodeReturn;
       pending = other.pending;
       pendingIndex = other.pendingIndex;
       network = other.network;
       state = other.state;
       action = other.action;
   
       #if ENS_VERSION_MAJOR >= 2
       updatePolicy = new typename UpdaterType::template
           Policy<arma::mat, arma::mat>(updater,
                                        network.Parameters().n_rows,
                                        network.Parameters().n_cols);
       #endif
   
       Reset();
   
       return *this;
     }
   
     OneStepSarsaWorker& operator=(OneStepSarsaWorker&& other)
     {
       if (&other == this)
         return *this;
   
       #if ENS_VERSION_MAJOR >= 2
       delete updatePolicy;
       #endif
   
       updater = std::move(other.updater);
       environment = std::move(other.environment);
       config = std::move(other.config);
       deterministic = std::move(other.deterministic);
       steps = std::move(other.steps);
       episodeReturn = std::move(other.episodeReturn);
       pending = std::move(other.pending);
       pendingIndex = std::move(other.pendingIndex);
       network = std::move(other.network);
       state = std::move(other.state);
       action = std::move(other.action);
   
       #if ENS_VERSION_MAJOR >= 2
       other.updatePolicy = NULL;
   
       updatePolicy = new typename UpdaterType::template
           Policy<arma::mat, arma::mat>(updater,
                                        network.Parameters().n_rows,
                                        network.Parameters().n_cols);
       #endif
   
       return *this;
     }
   
     ~OneStepSarsaWorker()
     {
       #if ENS_VERSION_MAJOR >= 2
       delete updatePolicy;
       #endif
     }
   
     void Initialize(NetworkType& learningNetwork)
     {
       #if ENS_VERSION_MAJOR == 1
       updater.Initialize(learningNetwork.Parameters().n_rows,
                          learningNetwork.Parameters().n_cols);
       #else
       delete updatePolicy;
   
       updatePolicy = new typename UpdaterType::template
           Policy<arma::mat, arma::mat>(updater,
                                        learningNetwork.Parameters().n_rows,
                                        learningNetwork.Parameters().n_cols);
       #endif
   
       // Build local network.
       network = learningNetwork;
     }
   
     bool Step(NetworkType& learningNetwork,
               NetworkType& targetNetwork,
               size_t& totalSteps,
               PolicyType& policy,
               double& totalReward)
     {
       // Interact with the environment.
       if (action.action == ActionType::size)
       {
         // Invalid action means we are at the beginning of an episode.
         arma::colvec actionValue;
         network.Predict(state.Encode(), actionValue);
         action = policy.Sample(actionValue, deterministic);
       }
       StateType nextState;
       double reward = environment.Sample(state, action, nextState);
       bool terminal = environment.IsTerminal(nextState);
       arma::colvec actionValue;
       network.Predict(nextState.Encode(), actionValue);
       ActionType nextAction = policy.Sample(actionValue, deterministic);
   
       episodeReturn += reward;
       steps++;
   
       terminal = terminal || steps >= config.StepLimit();
       if (deterministic)
       {
         if (terminal)
         {
           totalReward = episodeReturn;
           Reset();
           // Sync with latest learning network.
           network = learningNetwork;
           return true;
         }
         state = nextState;
         action = nextAction;
         return false;
       }
   
       #pragma omp atomic
       totalSteps++;
   
       pending[pendingIndex++] =
           std::make_tuple(state, action, reward, nextState, nextAction);
   
       if (terminal || pendingIndex >= config.UpdateInterval())
       {
         // Initialize the gradient storage.
         arma::mat totalGradients(learningNetwork.Parameters().n_rows,
             learningNetwork.Parameters().n_cols, arma::fill::zeros);
         for (size_t i = 0; i < pending.size(); ++i)
         {
           TransitionType &transition = pending[i];
   
           // Compute the target state-action value.
           arma::colvec actionValue;
           #pragma omp critical
           {
             targetNetwork.Predict(
                 std::get<3>(transition).Encode(), actionValue);
           };
           double targetActionValue = 0;
           if (!(terminal && i == pending.size() - 1))
             targetActionValue = actionValue[std::get<4>(transition).action];
           targetActionValue = std::get<2>(transition) +
               config.Discount() * targetActionValue;
   
           // Compute the training target for current state.
           arma::mat input = std::get<0>(transition).Encode();
           network.Forward(input, actionValue);
           actionValue[std::get<1>(transition).action] = targetActionValue;
   
           // Compute gradient.
           arma::mat gradients;
           network.Backward(input, actionValue, gradients);
   
           // Accumulate gradients.
           totalGradients += gradients;
         }
   
         // Clamp the accumulated gradients.
         totalGradients.transform(
             [&](double gradient)
             { return std::min(std::max(gradient, -config.GradientLimit()),
             config.GradientLimit()); });
   
         // Perform async update of the global network.
         #if ENS_VERSION_MAJOR == 1
         updater.Update(learningNetwork.Parameters(), config.StepSize(),
             totalGradients);
         #else
         updatePolicy->Update(learningNetwork.Parameters(),
             config.StepSize(), totalGradients);
         #endif
   
         // Sync the local network with the global network.
         network = learningNetwork;
   
         pendingIndex = 0;
       }
   
       // Update global target network.
       if (totalSteps % config.TargetNetworkSyncInterval() == 0)
       {
         #pragma omp critical
         { targetNetwork = learningNetwork; }
       }
   
       policy.Anneal();
   
       if (terminal)
       {
         totalReward = episodeReturn;
         Reset();
         return true;
       }
       state = nextState;
       action = nextAction;
       return false;
     }
   
    private:
     void Reset()
     {
       steps = 0;
       episodeReturn = 0;
       pendingIndex = 0;
       state = environment.InitialSample();
       using actions = typename EnvironmentType::Action::actions;
       action.action = static_cast<actions>(ActionType::size);
     }
   
     UpdaterType updater;
     #if ENS_VERSION_MAJOR >= 2
     typename UpdaterType::template Policy<arma::mat, arma::mat>* updatePolicy;
     #endif
   
     EnvironmentType environment;
   
     TrainingConfig config;
   
     bool deterministic;
   
     size_t steps;
   
     double episodeReturn;
   
     std::vector<TransitionType> pending;
   
     size_t pendingIndex;
   
     NetworkType network;
   
     StateType state;
   
     ActionType action;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
