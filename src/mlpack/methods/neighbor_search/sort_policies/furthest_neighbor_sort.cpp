/***
 * @file nearest_neighbor_sort.cpp
 * @author Ryan Curtin
 *
 * Implementation of the simple FurthestNeighborSort policy class.
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
#include "furthest_neighbor_sort.hpp"

using namespace mlpack::neighbor;

size_t FurthestNeighborSort::SortDistance(const arma::vec& list,
                                          const arma::Col<size_t>& indices,
                                          double newDistance)
{
  // The first element in the list is the nearest neighbor.  We only want to
  // insert if the new distance is greater than the last element in the list.
  if (newDistance < list[list.n_elem - 1])
    return (size_t() - 1); // Do not insert.

  // Search from the beginning.  This may not be the best way.
  for (size_t i = 0; i < list.n_elem; i++)
    if (newDistance >= list[i] || indices[i] == (size_t() - 1))
      return i;

  // Control should never reach here.
  return (size_t() - 1);
}
