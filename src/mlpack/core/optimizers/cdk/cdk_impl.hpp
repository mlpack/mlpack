/**
 * @file cdk_impl.hpp
 *
 * Implementation of stochastic gradient descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_OPTIMIZERS_CDK_CDK_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_CDK_CDK_IMPL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization /** Artificial Neural Network. */ {


template<typename RBMType>
CDK<RBMType>::CDK(RBMType& rbm, const size_t k, const double stepSize,
    const size_t maxIterations,
    const size_t batchSize,
    const bool shuffle,
    const bool persistent) :
    rbm(rbm),
    k(k),
    stepSize(stepSize),
    maxIterations(maxIterations),
    batchSize(batchSize),
    shuffle(shuffle),
    persistent(persistent)
    {
      // Nothing to do here
    }

template<typename RBMType>
void CDK<RBMType>::Optimize(arma::mat& iterate)
{
// Find the number of functions to use.
const size_t numFunctions = rbm.NumFunctions();

// This is used only if shuffle is true.
arma::Col<size_t> visitationOrder;
size_t currentFunction = 0;
arma::vec orderFunction(batchSize);

if (shuffle)
{
  visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t>>(0,
      (numFunctions - 1), numFunctions));
}


  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat cumgradient(iterate.n_rows, iterate.n_cols);

  for (size_t i = 1; i != maxIterations; ++i, ++currentFunction)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      if (shuffle) // Determine order of visitation.
        visitationOrder = arma::shuffle(visitationOrder);
      currentFunction = 0;
    }

    // Evaluate the gradient for this iteration.
    if (shuffle)
      rbm.Gradient(visitationOrder[currentFunction], k, persistent, gradient);
    else
      rbm.Gradient(currentFunction, k, persistent,  gradient);

    if (i % batchSize == 0)
    {
      std::cout << rbm.Evaluate(iterate, orderFunction) << std::endl;
      // Update the step
      iterate += stepSize * cumgradient;
      cumgradient.zeros();
    }
    else
    {
      cumgradient += gradient;
      if (shuffle)
        orderFunction(i % batchSize - 1) = visitationOrder[currentFunction];
      else
        orderFunction(i % batchSize - 1) = currentFunction;
    }
  }

  Log::Info << "CDK: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;
};
} // namespace optimization
} // namespace mlpack
#endif
