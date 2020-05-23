/**
 * @file sarsa_worker_transition_type.hpp
 * @author Arsen Zahray
 * this file contains definition of SarsaWorkerTransitionType class; SarsaWorkerTransitionType is data type used by Worker class to store training data
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/methods/reinforcement_learning/worker/q_learning_worker_transition_type.hpp>
#ifndef SARSA_WORKER_TRANSITION_TYPE
#define SARSA_WORKER_TRANSITION_TYPE
namespace mlpack {
namespace rl {
template<
    typename EnvironmentType
>
struct SarsaWorkerTransitionType:
        public QLearningWorkerTransitionType<
                typename EnvironmentType::State,
                typename EnvironmentType::Action>
{
    using base=QLearningWorkerTransitionType<
            typename EnvironmentType::State,
            typename EnvironmentType::Action>;
    using StateType = typename EnvironmentType::State;
    using ActionType = typename EnvironmentType::Action;

    ActionType nextAction;

    SarsaWorkerTransitionType() = default;

    SarsaWorkerTransitionType(
            const SarsaWorkerTransitionType&) = default;

    SarsaWorkerTransitionType(
            SarsaWorkerTransitionType&&) = default;

    SarsaWorkerTransitionType& operator =
            (const SarsaWorkerTransitionType&) = default;

    SarsaWorkerTransitionType& operator =
            (SarsaWorkerTransitionType&&) = default;

    SarsaWorkerTransitionType(
            const StateType &state,
            const ActionType &action,
            const double reward,
            const StateType &nextState,
            const ActionType &nextAction):
        base(state,action,reward,nextState),
        nextAction(nextAction)
    {  }

    SarsaWorkerTransitionType(
            const std::tuple<
                    StateType,
                    ActionType,
                    double,
                    StateType,
                    ActionType> &tp):
        base(tp),
        nextAction(std::get<4>(tp))
    {  }
};
} // namespace rl
} // namespace mlpack
#endif // !Q_LEARNING_WORKER_TRANSITION_TYPE


