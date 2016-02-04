/***
 * @file nearest_neighbor_sort.cpp
 * @author Ryan Curtin
 *
 * Implementation of the simple FurthestNeighborSort policy class.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
