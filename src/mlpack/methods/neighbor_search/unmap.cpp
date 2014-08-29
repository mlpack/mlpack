/**
 * @file unmap.cpp
 * @author Ryan Curtin
 *
 * Auxiliary function to unmap neighbor search results.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "unmap.hpp"

namespace mlpack {
namespace neighbor {

// Useful in the dual-tree setting.
void Unmap(const arma::Mat<size_t>& neighbors,
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
void Unmap(const arma::Mat<size_t>& neighbors,
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

}; // namespace neighbor
}; // namespace mlpack
