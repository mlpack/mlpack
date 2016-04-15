/**
 * @file ra_typedef.hpp
 * @author Parikshit Ram
 *
 * Simple typedefs describing template instantiations of the RASearch
 * class which are commonly used.
 */
#ifndef MLPACK_RANN_RA_TYPEDEF_HPP
#define MLPACK_RANN_RA_TYPEDEF_HPP

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

} // namespace neighbor
} // namespace mlpack

#endif
