/**
 * @file smorms3_impl.hpp
 * @author Vivek Pal
 *
 * Implementation of the SMORMS3 optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SMORMS3_SMORMS3_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_SMORMS3_SMORMS3_IMPL_HPP

// In case it hasn't been included yet.
#include "smorms3.hpp"

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
SMORMS3<DecomposableFunctionType>::SMORMS3(DecomposableFunctionType& function,
                                     const double lRate,
                                     const double eps,
                                     const size_t maxIterations,
                                     const double tolerance,
                                     const bool shuffle) :
    function(function),
    lRate(lRate),
    eps(eps),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double SMORMS3<DecomposableFunctionType>::Optimize(arma::mat& iterate)
{
  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // This is used only if shuffle is true.
  arma::Col<size_t> visitationOrder;
  if (shuffle)
    visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t>>(0,
        (numFunctions - 1), numFunctions));

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  // Initialise the parameters mem, g and g2.
  arma::mat mem = arma::ones<arma::mat>(iterate.n_rows, iterate.n_cols);

  arma::mat g = arma::zeros<arma::mat>(iterate.n_rows, iterate.n_cols);

  arma::mat g2 = arma::zeros<arma::mat>(iterate.n_rows, iterate.n_cols);

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  
  for (size_t i = 1; i != maxIterations; ++i, ++currentFunction)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      // Output current objective function.
      Log::Info << "SMORMS3: iteration " << i << ", objective "
          << overallObjective << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "SMORMS3: converged to " << overallObjective
            << "; terminating with failure. Try a smaller step size?"
            << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "SMORMS3: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;
        return overallObjective;
      }

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;

      if (shuffle) // Determine order of visitation.
        visitationOrder = arma::shuffle(visitationOrder);
    }

    // Evaluate the gradient for this iteration.
    if (shuffle)
      function.Gradient(iterate, visitationOrder[currentFunction], gradient);
    else
      function.Gradient(iterate, currentFunction, gradient);

    // And update the iterate.
    arma::mat r = 1 / (mem + 1);

    g = (1 - r) % g;
    g += r % gradient;

    g2 = (1 - r) % g2;
    g2 += r % (gradient % gradient);

    arma::mat x = (g % g) / (g2 + eps);

    arma::mat lRateMat(x.n_rows, x.n_cols);
    lRateMat.fill(lRate);

    iterate -= gradient * arma::min(x, lRateMat) / (arma::sqrt(g2) + eps);

    mem *= (1 - x);
    mem += 1;

    // Now add that to the overall objective function.
    if (shuffle)
      overallObjective += function.Evaluate(iterate,
          visitationOrder[currentFunction]);
    else
      overallObjective += function.Evaluate(iterate, currentFunction);
  }

  Log::Info << "SMORMS3: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;
  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
