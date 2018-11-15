/**
 * @file grid_search_impl.hpp
 * @author Kirill Mishchenko
 *
 * Implementation of the grid-search optimization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_GRID_SEARCH_GRID_SEARCH_IMPL_HPP
#define ENSMALLEN_GRID_SEARCH_GRID_SEARCH_IMPL_HPP

#include <limits>
#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename FunctionType>
double GridSearch::Optimize(
    FunctionType& function,
    arma::mat& bestParameters,
    const std::vector<bool>& categoricalDimensions,
    const arma::Row<size_t>& numCategories)
{
  if (categoricalDimensions.size() != iterate.n_rows)
  {
    std::ostringstream oss;
    oss << "GridSearch::Optimize(): expected information about "
        << iterate.n_rows << " dimensions in categoricalDimensions, "
        << "but got " << categoricalDimensions.size();
    throw std::invalid_argument(oss.str());
  }

  if (numCategories.n_elem != iterate.n_rows)
  {
    std::ostringstream oss;
    oss << "GridSearch::Optimize(): expected numCategories to have length "
        << "equal to number of dimensions (" << iterate.n_rows << ") but it has"
        << " length " << numCategories.n_elem;
    throw std::invalid_argument(oss.str());
  }

  for (size_t i = 0; i < categoricalDimensions.size(); ++i)
  {
    if (categoricalDimensions[i])
    {
      std::ostringstream oss;
      oss << "GridSearch::Optimize(): the dimension " << i
          << "is not categorical" << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }

  double bestObjective = std::numeric_limits<double>::max();
  bestParameters = arma::mat(categoricalDimensions.size(), 1);
  arma::vec currentParameters = arma::vec(categoricalDimensions.size());

  /* Initialize best parameters for the case (very unlikely though) when no set
   * of parameters gives an objective value better than
   * std::numeric_limits<double>::max() */
  for (size_t i = 0; i < categoricalDimensions.size(); ++i)
    bestParameters(i, 0) = 0;

  Optimize(function, bestObjective, bestParameters, currentParameters,
      categoricalDimensions, numCategories, 0);

  return bestObjective;
}

template<typename FunctionType>
void GridSearch::Optimize(
    FunctionType& function,
    double& bestObjective,
    arma::mat& bestParameters,
    arma::vec& currentParameters,
    const std::vector<bool>& categoricalDimensions,
    const arma::Row<size_t>& numCategories,
    size_t i)
{
  // Make sure we have the methods that we need.
  traits::CheckNonDifferentiableFunctionTypeAPI<FunctionType>();

  if (i < categoricalDimensions.size())
  {
    for (size_t j = 0; j < numCategories(i); ++j)
    {
      currentParameters(i) = j;
      Optimize(function, bestObjective, bestParameters, currentParameters,
          categoricalDimensions, numCategories, i + 1);
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

} // namespace ens

#endif
