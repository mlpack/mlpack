/**
 * @file methods/reinforcement_learning/policy/exponential_policy.hpp
 * @author Nanubala Gnana Sai
 *
 * This file is the implementation of ExponentialPolicy class.
 * An Exponential policy would increase the lambda value exponentially.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_POLICY_EXPONENTIAL_POLICY_HPP
#define MLPACK_METHODS_RL_POLICY_EXPONENTIAL_POLICY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/dists/discrete_distribution.hpp>

namespace mlpack {
namespace rl {

/**
 *  Exponential lambda update policy.
 */
class ExponentialPolicy
{
 public:

  /**
   * @param lambdaInit The starting value of lambda.
   * @param lambda Parameter controlling the loss surface.
   * @param delta Increment value of lambda.
   * @param expBase Base exponential for incrementing lambda.
   */
  ExponentialPolicy(double lambdaInit = 0.01,
                    double lambda = 0.01,
                    double delta = 0,
                    double expBase = 1):
                  lambdaInit(lambdaInit),
                  lambda(lambda),
                  delta(delta),
                  expBase(expBase)
  { /* Nothing to do here. */ };

  void Accumulate()
  {
    lambda += delta;
    delta = (lambda - lambdaInit) * expBase + lambdaInit - lambda;
  }

  //! Get the starting value of lambda.
  double LambdaInit() const { return lambdaInit; }
  //! Modify the starting value of lambda.
  double& LambdaInit() { return lambdaInit; }

  //! Get the current value of lambda.
  double Lambda() const { return lambda; }
  //! Modify the current value of lambda.
  double& Lambda() { return lambda; }

  //! Get the current value of delta.
  double Delta() const { return delta; }
  //! Modify the current value of delta.
  double& Delta() { return delta; }

  //! Get the current value of exponential base.
  double ExpBase() const { return expBase; }
  //! Modify the current value of exponential base.
  double& ExpBase() { return expBase; }

 private:
    //! Locally-stored flag for the lambdaInit value.
    double lambdaInit;

    //! Locally-stored flag for the lambda value.
    double lambda;

    //! Locally-stored flag for the delta value.
    double delta;

    //! Locally-stored flag for the expBase value.
    double expBase;
};

} // namespace rl
} // namespace mlpack

#endif
