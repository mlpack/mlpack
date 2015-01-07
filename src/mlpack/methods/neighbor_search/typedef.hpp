/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Simple typedefs describing template instantiations of the NeighborSearch
 * class which are commonly used.  This is meant to be included by
 * neighbor_search.h but is a separate file for simplicity.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
