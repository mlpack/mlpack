/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Simple typedefs describing template instantiations of the NeighborSearch
 * class which are commonly used.  This is meant to be included by
 * neighbor_search.h but is a separate file for simplicity.
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_NEIGHBOR_SEARCH_TYPEDEF_H
#define __MLPACK_NEIGHBOR_SEARCH_TYPEDEF_H

// In case someone included this directly.
#include "neighbor_search.hpp"

#include <mlpack/core/metrics/lmetric.hpp>

#include "sort_policies/nearest_neighbor_sort.hpp"
#include "sort_policies/furthest_neighbor_sort.hpp"

namespace mlpack {
namespace neighbor {

/**
 * The AllkNN class is the all-k-nearest-neighbors method.  It returns L2
 * distances (Euclidean distances) for each of the k nearest neighbors.
 */
typedef NeighborSearch<NearestNeighborSort, metric::EuclideanDistance> AllkNN;

/**
 * The AllkFN class is the all-k-furthest-neighbors method.  It returns L2
 * distances (Euclidean distances) for each of the k furthest neighbors.
 */
typedef NeighborSearch<FurthestNeighborSort, metric::EuclideanDistance> AllkFN;

}; // namespace neighbor
}; // namespace mlpack

#endif
