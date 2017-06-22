/**
 * @file parallel_sgd.hpp
 * @author Shikhar Bhardwaj
 *
 * Parallel Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_HPP

#include <mlpack/prereqs.hpp>

#include "stepsize_policies/constant_step.hpp"

// TODO FIXME : Documentation

namespace mlpack {
namespace optimization {

/**
 * An implementation of parallel stochastic gradient descent using the lock-free
 * HOGWILD! approach.
 *
 * For more information, see the following.
 * @misc{1106.5730,
 *   Author = {Feng Niu and Benjamin Recht and Christopher Re and Stephen J.
 *             Wright},
 *   Title = {HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient
 *            Descent},
 *   Year = {2011},
 *   Eprint = {arXiv:1106.5730},
 * }
 */
template <
  typename SparseFunctionType,
  typename StepsizePolicyType = ConstantStep
>
class ParallelSGD
{
 public:
  ParallelSGD(SparseFunctionType& function, 
              const size_t maxIterations = 100000, 
              const double tolerance = 1e-5,
              const StepsizePolicyType stepPolicy = StepsizePolicyType());

  double Optimize(SparseFunctionType& function, arma::mat& iterate);

  double Optimize(arma::mat& iterate)
  {
    return Optimize(this->function, iterate);
  }

  //! Get the instantiated function to be optimized.
  const SparseFunctionType& Function() const { return function; }
  //! Modify the instantiated function.
  SparseFunctionType& Function() { return function; }
  //! Get the instantiated function to be optimized.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the instantiated function.
  size_t& MaxIterations() { return maxIterations; }
  //! Get the instantiated function to be optimized.
  double Tolerance() const { return tolerance; }
  //! Modify the instantiated function.
  double& Tolerance() { return tolerance; }

 private:
  SparseFunctionType& function;
  size_t maxIterations;
  double tolerance;
  StepsizePolicyType& stepPolicy;
};

}
}

// Include implementation.
#include "parallel_sgd_impl.hpp"

#endif
