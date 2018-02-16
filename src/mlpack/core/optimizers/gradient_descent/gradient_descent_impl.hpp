/**
 * @file gradient_descent_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Simple gradient descent implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_GRADIENT_DESCENT_GRADIENT_DESCENT_IMPL_HPP

// In case it hasn't been included yet.
#include "gradient_descent.hpp"

#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

//! Optimize the function (minimize).
template<typename FunctionType>
double GradientDescent::Optimize(
    FunctionType& function, arma::mat& iterate)
{
  // Use the Function<> wrapper type to provide additional functionality.
  typedef Function<FunctionType> FullFunctionType;
  FullFunctionType& f(static_cast<FullFunctionType&>(function));

  // Make sure we have the methods that we need.
  traits::CheckFunctionTypeAPI<FullFunctionType>();

  // To keep track of where we are and how things are going.
  double overallObjective = std::numeric_limits<double>::max();
  double lastObjective = std::numeric_limits<double>::max();

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  for (size_t i = 1; i != maxIterations; ++i)
  {
    overallObjective = f.EvaluateWithGradient(iterate, gradient);

    // Output current objective function.
    Log::Info << "Gradient Descent: iteration " << i << ", objective "
        << overallObjective << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Log::Warn << "Gradient Descent: converged to " << overallObjective
          << "; terminating" << " with failure.  Try a smaller step size?"
          << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Log::Info << "Gradient Descent: minimized within tolerance "
          << tolerance << "; " << "terminating optimization." << std::endl;
      return overallObjective;
    }

    // Reset the counter variables.
    lastObjective = overallObjective;

    // And update the iterate.
    iterate -= stepSize * gradient;
  }

  Log::Info << "Gradient Descent: maximum iterations (" << maxIterations
      << ") reached; " << "terminating optimization." << std::endl;
  return overallObjective;
}

template<typename FunctionType>
double GradientDescent::Optimize(
    FunctionType& function,
    arma::mat& iterate,
    data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo)
{
  if (datasetInfo.Dimensionality() != iterate.n_rows)
  {
      std::ostringstream oss;
      oss << "GradientDescent::Optimize(): expected information about "
          << iterate.n_rows << " dimensions in datasetInfo, but found about "
          << datasetInfo.Dimensionality() << std::endl;
      throw std::invalid_argument(oss.str());
  }

  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    if (datasetInfo.Type(i) != data::Datatype::numeric)
    {
      std::ostringstream oss;
      oss << "GradientDescent::Optimize(): the dimension " << i
          << "is not numeric" << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }

  return Optimize(function, iterate);
}

} // namespace optimization
} // namespace mlpack

#endif
