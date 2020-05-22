/**
 * @file greedy_policy.hpp
 * @author Shangtong Zhang
 * @author Abhinav Sagar
 * @author Arsen Zahray
 *
 * This file is an implementation of epsilon greedy policy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef CONTINUOUS_GDEEDY_POLICY
#define CONTINUOUS_GDEEDY_POLICY

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

namespace mlpack {
    namespace rl {
        template <typename EnvironmentType>
        class ContinuousGreedyPolicy
        {
        public:
            //! Convenient typedef for action.
            using ActionType = typename EnvironmentType::Action;

            /**
           * Constructor for continuous epsilon greedy policy class.
           *
           * @param initialEpsilon The initial probability to explore
           *        (select a random action).
           * @param minEpsilon Epsilon will never be less than this value.
           * @param decayRate at each step, probability of selecting random action will decrease by `1-decayRate`
           */
            ContinuousGreedyPolicy(const double initialEpsilon,
                                              const double minEpsilon,
                                              const double decayRate = 1e-10) :
                    epsilon(initialEpsilon),
                    minEpsilon(minEpsilon),
                    delta(1-decayRate)
            { /* Nothing to do here. */
            }

            /**
             * Sample an action based on given action values.
             *
             * @param actionValue Values for each action.
             * @param deterministic Always select the action greedily.
             * @return Sampled action.
             */
            ActionType Sample(const arma::colvec& actionValue, bool deterministic = false)
            {
                // Select the action randomly.
                if (!deterministic)
                {
                    double exploration = math::Random();
                    if (exploration < epsilon)
                    {
                        return ActionType::Sample();
                    }
                }

                // Select the action greedily.
                return ActionType(actionValue);
            }

            /**
             * Exploration probability will anneal at each step.
             */
            void Anneal()
            {
                epsilon *= delta;
                epsilon = std::max(minEpsilon, epsilon);
            }

            /**
             * @return Current possibility to explore.
             */
            const double& Epsilon() const { return epsilon; }

        private:
            //! Locally-stored probability to explore.
            double epsilon;

            //! Locally-stored lower bound for epsilon.
            double minEpsilon;

            //! Locally-stored stride for epsilon to anneal.
            double delta;
        };
    }
}
#endif