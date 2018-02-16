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
   * Optimize (minimize) the given function by iterating through the all
   * possible combinations of values for the parameters specified in
   * datasetInfo.
   *
   * @param function Function to optimize.
   * @param bestParameters Variable for storing results.
   * @param datasetInfo Type information for each dimension of the dataset. It
   *     should store possible values for each parameter.
   * @return Objective value of the final point.
   */
  template<typename FunctionType>
  double Optimize(
      FunctionType& function,
      arma::mat& bestParameters,
      data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo);

 private:
  /**
   * Iterate through the last (parameterValueCollections.size() - i) dimensions
   * of the grid and change the arguments bestObjective and bestParameters if
   * there is something better. The values for the first i dimensions
   * (parameters) are specified in the first i rows of the currentParameters
   * argument.
   */
  template<typename FunctionType>
  void Optimize(
      FunctionType& function,
      double& bestObjective,
      arma::mat& bestParameters,
      arma::vec& currentParameters,
      data::DatasetMapper<data::IncrementPolicy, double>& datasetInfo,
      size_t i);
};

} // namespace optimization
} // namespace mlpack

// Include implementation
#include "grid_search_impl.hpp"

#endif
