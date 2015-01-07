/**
 * @file ra_typedef.hpp
 * @author Parikshit Ram
 *
 * Simple typedefs describing template instantiations of the RASearch
 * class which are commonly used.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_RANN_RA_TYPEDEF_HPP
#define __MLPACK_RANN_RA_TYPEDEF_HPP

// In case someone included this directly.
#include "ra_search.hpp"

#include <mlpack/core/metrics/lmetric.hpp>

#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/furthest_neighbor_sort.hpp>

namespace mlpack {
namespace neighbor {

/**
 * The AllkRANN class is the all-k-rank-approximate-nearest-neighbors method.
 * It returns squared L2 distances (squared Euclidean distances) for each of the
 * k rank-approximate nearest-neighbors.  Squared distances are used because
 * they are slightly faster than non-squared distances (they have one fewer call
 * to sqrt()).
 *
 * The approximation is controlled with two parameters (see allkrann_main.cpp)
 * which can be specified at search time. So the tree building is done only once
 * while the search can be performed multiple times with different approximation
 * levels.
 */
typedef RASearch<> AllkRANN;

/**
 * The AllkRAFN class is the all-k-rank-approximate-farthest-neighbors method.
 * It returns squared L2 distances (squared Euclidean distances) for each of the
 * k rank-approximate farthest-neighbors.  Squared distances are used because
 * they are slightly faster than  non-squared distances (they have one fewer
 * call to sqrt()).
 *
 * The approximation is controlled with two parameters (see allkrann_main.cpp)
 * which can be specified at search time. So the tree building is done only once
 * while the search can be performed multiple times with different approximation
 * levels.
 */
typedef RASearch<FurthestNeighborSort> AllkRAFN;

}; // namespace neighbor
}; // namespace mlpack

#endif
