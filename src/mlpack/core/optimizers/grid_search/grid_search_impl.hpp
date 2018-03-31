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
#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

template<typename FunctionType>
double GridSearch::Optimize(
    FunctionType& function,
    arma::mat& bestParameters,
    data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo)
{
  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    if (datasetInfo.Type(i) != data::Datatype::categorical)
    {
      std::ostringstream oss;
      oss << "GridSearch::Optimize(): the dimension " << i
          << "is not categorical" << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }

  double bestObjective = std::numeric_limits<double>::max();
  bestParameters = arma::mat(datasetInfo.Dimensionality(), 1);
  arma::vec currentParameters = arma::vec(datasetInfo.Dimensionality());

  /* Initialize best parameters for the case (very unlikely though) when no set
   * of parameters gives an objective value better than
   * std::numeric_limits<double>::max() */
  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
    bestParameters(i, 0) = datasetInfo.UnmapString(0, i);

  Optimize(function, bestObjective, bestParameters, currentParameters,
      datasetInfo, 0);

  return bestObjective;
}

template<typename FunctionType>
void GridSearch::Optimize(
    FunctionType& function,
    double& bestObjective,
    arma::mat& bestParameters,
    arma::vec& currentParameters,
    data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
    size_t i)
{
  // Make sure we have the methods that we need.
  traits::CheckNonDifferentiableFunctionTypeAPI<FunctionType>();

  if (i < datasetInfo.Dimensionality())
  {
    for (size_t j = 0; j < datasetInfo.NumMappings(i); ++j)
    {
      currentParameters(i) = datasetInfo.UnmapString(j, i);
      Optimize(function, bestObjective, bestParameters, currentParameters,
          datasetInfo, i + 1);
    }
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
