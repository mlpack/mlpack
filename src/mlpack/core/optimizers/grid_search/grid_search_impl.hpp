/**
 * @file grid_search_impl.hpp
 * @author Kirill Mishchenko
 *
 * Implementation of the grid-search optimization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_GRID_SEARCH_GRID_SEARCH_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_GRID_SEARCH_GRID_SEARCH_IMPL_HPP

#include <limits>

namespace mlpack {
namespace optimization {

template<typename... Collections>
GridSearch::GridSearch(const Collections&... collections)
{
  InitParameterValueCollections(collections...);
}

template<typename FunctionType>
double GridSearch::Optimize(FunctionType& function, arma::mat& bestParameters)
{
  double bestObjective = std::numeric_limits<double>::max();
  bestParameters = arma::mat(parameterValueCollections.size(), 1);
  arma::vec currentParameters = arma::vec(parameterValueCollections.size());

  /* Initialize best parameters for the case (very unlikely though) when no set
   * of parameters gives an objective value better than
   * std::numeric_limits<double>::max() */
  for (size_t i = 0; i < parameterValueCollections.size(); ++i)
    bestParameters(i, 0) = parameterValueCollections[i][0];

  Optimize(function, bestObjective, bestParameters, currentParameters, 0);

  return bestObjective;
}

template<typename FunctionType>
void GridSearch::Optimize(FunctionType& function,
                          double& bestObjective,
                          arma::mat& bestParameters,
                          arma::vec& currentParameters,
                          size_t i)
{
  if (i < parameterValueCollections.size())
    for (double value : parameterValueCollections[i])
    {
      currentParameters(i) = value;
      Optimize(function, bestObjective, bestParameters, currentParameters,
          i + 1);
    }
  else
  {
    double objective = function.Evaluate(currentParameters);
    if (objective < bestObjective)
    {
      bestObjective = objective;
      bestParameters = currentParameters;
    }
  }
}

} // namespace optimization
} // namespace mlpack

#endif
