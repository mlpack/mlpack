/**
 * @file grid_search.hpp
 * @author Kirill Mishchenko
 *
 * Grid-search optimization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_GRID_SEARCH_GRID_SEARCH_HPP
#define MLPACK_CORE_OPTIMIZERS_GRID_SEARCH_GRID_SEARCH_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

/**
 * An optimizer that finds the minimum of a given function by iterating through
 * points on a multidimensional grid.
 *
 * For GridSearch to work, a FunctionType template parameter is required. This
 * class must implement the following function:
 *
 *   double Evaluate(const arma::mat& coordinates);
 */
class GridSearch
{
 public:
  /**
   * Initialize a GridSearch object.
   *
   * @param collections Collections of values (one for each parameter). Each
   *     collection should be an STL-compatible container (it should provide
   *     begin() and end() methods returning iterators).
   */
  template<typename... Collections>
  GridSearch(const Collections&... collections);

  /**
   * Optimize (minimize) the given function by iterating through the all
   * possible combinations of values for the parameters.
   */
  template<typename FunctionType>
  double Optimize(FunctionType& function, arma::mat& bestParameters);

 private:
  //! Collections of parameter values (one for each parameter).
  std::vector<std::vector<double>> parameterValueCollections;

  /**
   * Iterate through the last (parameterValueCollections.size() - i) dimensions
   * of the grid and change the arguments bestObjective and bestParameters if
   * there is something better. The values for the first i dimensions
   * (parameters) are specified in the first i rows of the currentParameters
   * argument.
   */
  template<typename FunctionType>
  void Optimize(FunctionType& function,
                double& bestObjective,
                arma::mat& bestParameters,
                arma::vec& currentParameters,
                size_t i);

  /**
   * Convert each specified collection into a std::vector<double> container and
   * put results into parameterValueCollections.
   */
  template<typename Collection, typename... Collections>
  void InitParameterValueCollections(const Collection& collection,
                                     const Collections&... collections)
  {
    parameterValueCollections.push_back(
        std::vector<double>(collection.begin(), collection.end()));

    InitParameterValueCollections(collections...);
  }

  /**
   * Finish initialization.
   */
  void InitParameterValueCollections() {}
};

} // namespace optimization
} // namespace mlpack

// Include implementation
#include "grid_search_impl.hpp"

#endif
