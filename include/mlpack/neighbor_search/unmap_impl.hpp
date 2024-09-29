/**
 * @file methods/neighbor_search/unmap_impl.hpp
 * @author Ryan Curtin
 *
 * Auxiliary function to unmap neighbor search results.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_UNMAP_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_UNMAP_IMPL_HPP

#include "unmap.hpp"

namespace mlpack {

// Useful in the dual-tree setting.
inline void Unmap(const arma::Mat<size_t>& neighbors,
                  const arma::mat& distances,
                  const std::vector<size_t>& referenceMap,
                  const std::vector<size_t>& queryMap,
                  arma::Mat<size_t>& neighborsOut,
                  arma::mat& distancesOut,
                  const bool squareRoot)
{
  // Set matrices to correct size.
  neighborsOut.set_size(neighbors.n_rows, neighbors.n_cols);
  distancesOut.set_size(distances.n_rows, distances.n_cols);

  // Unmap distances.
  for (size_t i = 0; i < distances.n_cols; ++i)
  {
    // Map columns to the correct place.  The ternary operator does not work
    // here...
    if (squareRoot)
      distancesOut.col(queryMap[i]) = sqrt(distances.col(i));
    else
      distancesOut.col(queryMap[i]) = distances.col(i);

    // Map indices of neighbors.
    for (size_t j = 0; j < distances.n_rows; ++j)
      neighborsOut(j, queryMap[i]) = referenceMap[neighbors(j, i)];
  }
}

// Useful in the single-tree setting.
inline void Unmap(const arma::Mat<size_t>& neighbors,
                  const arma::mat& distances,
                  const std::vector<size_t>& referenceMap,
                  arma::Mat<size_t>& neighborsOut,
                  arma::mat& distancesOut,
                  const bool squareRoot)
{
  // Set matrices to correct size.
  neighborsOut.set_size(neighbors.n_rows, neighbors.n_cols);

  // Take square root of distances, if necessary.
  if (squareRoot)
    distancesOut = sqrt(distances);
  else
    distancesOut = distances;

  // Map neighbors back to original locations.
  for (size_t j = 0; j < neighbors.n_elem; ++j)
    neighborsOut[j] = referenceMap[neighbors[j]];
}

} // namespace mlpack

#endif
