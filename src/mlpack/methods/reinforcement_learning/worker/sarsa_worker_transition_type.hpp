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

#ifndef SARSA_WORKER_TRANSITION_TYPE
#define SARSA_WORKER_TRANSITION_TYPE
namespace mlpack {
    namespace rl {
        template<
            typename EnvironmentType
        >
        struct SarsaWorkerTransitionType
        {
            using StateType = typename EnvironmentType::State;
            using ActionType = typename EnvironmentType::Action;
        
            StateType state;
            ActionType action, nextAction;
            double reward;
            StateType nextState;

            SarsaWorkerTransitionType() = default;
        	

            SarsaWorkerTransitionType(StateType state, ActionType action, double reward, StateType nextState, ActionType nextAction)
                :state(state), action(action), reward(reward), nextState(nextState), nextAction(nextAction)
            {  }

            SarsaWorkerTransitionType(std::tuple<StateType, ActionType, double, StateType, ActionType> tp)
                :state(std::get<0>(tp)), action(std::get<1>(tp)), reward(std::get<2>(tp)), nextState(std::get<3>(tp)), nextAction(std::get<4>(tp))
            {  }
        };
    }
}
#endif // !Q_LEARNING_WORKER_TRANSITION_TYPE


