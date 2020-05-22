/**
 * @file worker_base.hpp
 * @author Shangtong Zhang
 * @author Arsen Zahray
 * this file contains definition of WorkerBase class; WorkerBase provides structure on which all workers will be built
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef WORKER_BASE
#define WORKER_BASE

#include <mlpack/methods/reinforcement_learning/training_config.hpp>

namespace mlpack {
    namespace rl {

        /**
         * This class is responsible for resource management of workers
         *
         * @tparam EnvironmentType The type of the reinforcement learning task.
         * @tparam NetworkType The type of the network model.
         * @tparam UpdaterType The type of the optimizer.
         * @tparam PolicyType The type of the behavior policy. *
         */
        template <
            typename EnvironmentType,
            typename NetworkType,
            typename UpdaterType,
            typename PolicyType,
    		typename TransitionType
        >
        class WorkerBase
        {
        public:
            using StateType = typename EnvironmentType::State;
            using ActionType = typename EnvironmentType::Action;
            //using TransitionType = typename Subclass::TransitionType;

            /**
             * Construct one step Q-Learning worker with the given parameters and
             * environment.
             *
             * @param updater The optimizer.
             * @param environment The reinforcement learning task.
             * @param config Hyper-parameters.
             * @param deterministic Whether it should be deterministic.
             */
            WorkerBase(
                const UpdaterType& updater,
                const EnvironmentType& environment,
                const TrainingConfig& config,
                bool deterministic) :
                updater(updater),
#if ENS_VERSION_MAJOR >= 2
                updatePolicy(NULL),
#endif
                environment(environment),
                config(config),
                deterministic(deterministic),
                pending(config.UpdateInterval())
            {
                Reset();
            }

            /**
             * Copy another OneStepQLearningWorker.
             *
             * @param other OneStepQLearningWorker to copy.
             */
            WorkerBase(const WorkerBase& other) :
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
                state(other.state)
            {
#if ENS_VERSION_MAJOR >= 2
                updatePolicy = new typename UpdaterType::template
                    Policy<arma::mat, arma::mat>(updater,
                        network.Parameters().n_rows,
                        network.Parameters().n_cols);
#endif

                Reset();
            }

            /**
             * Take ownership of another OneStepQLearningWorker.
             *
             * @param other OneStepQLearningWorker to take ownership of.
             */
            WorkerBase(WorkerBase&& other) :
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
                state(std::move(other.state))
            {
#if ENS_VERSION_MAJOR >= 2
                other.updatePolicy = NULL;

                updatePolicy = new typename UpdaterType::template
                    Policy<arma::mat, arma::mat>(updater,
                        network.Parameters().n_rows,
                        network.Parameters().n_cols);
#endif
            }

            /**
             * Copy another OneStepQLearningWorker.
             *
             * @param other OneStepQLearningWorker to copy.
             */
            WorkerBase& operator=(const WorkerBase& other)
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

#if ENS_VERSION_MAJOR >= 2
                updatePolicy = new typename UpdaterType::template
                    Policy<arma::mat, arma::mat>(updater,
                        network.Parameters().n_rows,
                        network.Parameters().n_cols);
#endif

                Reset();

                return *this;
            }

            /**
             * Take ownership of another OneStepQLearningWorker.
             *
             * @param other OneStepQLearningWorker to take ownership of.
             */
            WorkerBase& operator=(WorkerBase&& other)
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

#if ENS_VERSION_MAJOR >= 2
                other.updatePolicy = NULL;

                updatePolicy = new typename UpdaterType::template
                    Policy<arma::mat, arma::mat>(updater,
                        network.Parameters().n_rows,
                        network.Parameters().n_cols);
#endif

                return *this;
            }

            /**
             * Clean memory.
             */
            ~WorkerBase()
            {
#if ENS_VERSION_MAJOR >= 2
                delete updatePolicy;
#endif
            }

            /**
             * Initialize the worker.
             * @param learningNetwork The shared network.
             */
            virtual void Initialize(NetworkType& learningNetwork)
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

            /**
             * The agent will execute one step.
             *
             * @param learningNetwork The shared learning network.
             * @param targetNetwork The shared target network.
             * @param totalSteps The shared counter for total steps.
             * @param policy The shared behavior policy.
             * @param totalReward This will be the episode return if the episode ends
             *     after this step. Otherwise this is invalid.
             * @return Indicate whether current episode ends after this step.
             */
            virtual bool Step(NetworkType& learningNetwork,
                NetworkType& targetNetwork,
                size_t& totalSteps,
                PolicyType& policy,
                double& totalReward) = 0;
        protected:
            /**
             * Reset the worker for a new episode.
             */
            virtual void Reset()
            {
                steps = 0;
                episodeReturn = 0;
                pendingIndex = 0;
                state = environment.InitialSample();
            }

            //! Locally-stored optimizer.
            UpdaterType updater;
#if ENS_VERSION_MAJOR >= 2
            typename UpdaterType::template Policy<arma::mat, arma::mat>* updatePolicy;
#endif

            //! Locally-stored task.
            EnvironmentType environment;

            //! Locally-stored hyper-parameters.
            TrainingConfig config;

            //! Whether this episode is deterministic or not.
            bool deterministic;

            //! Total steps in current episode.
            size_t steps;

            //! Total reward in current episode.
            double episodeReturn;

            //! Buffer for delayed update.
            std::vector<TransitionType> pending;

            //! Current position of the buffer.
            size_t pendingIndex;

            //! Local network of the worker.
            NetworkType network;

            //! Current state of the agent.
            StateType state;
        };

    } // namespace rl
} // namespace mlpack

#endif
