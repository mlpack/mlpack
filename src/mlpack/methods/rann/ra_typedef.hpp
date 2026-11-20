/**
 * @file methods/rann/ra_typedef.hpp
 * @author Parikshit Ram
 *
 * Simple typedefs describing template instantiations of the RASearch
 * class which are commonly used.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_RANN_RA_TYPEDEF_HPP
#define MLPACK_RANN_RA_TYPEDEF_HPP

// In case someone included this directly.
#include "ra_search.hpp"

#include <mlpack/core/distances/lmetric.hpp>

#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/furthest_neighbor_sort.hpp>

namespace mlpack {

/**
 * The KRANN class is the k-rank-approximate-nearest-neighbors method.  It
 * returns L2 distances for each of the k rank-approximate nearest-neighbors.
 *
 * The approximation is controlled with two parameters (see allkrann_main.cpp)
 * which can be specified at search time. So the tree building is done only once
 * while the search can be performed multiple times with different approximation
 * levels.
 */
using KRANN = RASearch<>;

/**
 * The KRAFN class is the k-rank-approximate-farthest-neighbors method.  It
 * returns L2 distances for each of the k rank-approximate farthest-neighbors.
 *
 * The approximation is controlled with two parameters (see allkrann_main.cpp)
 * which can be specified at search time. So the tree building is done only once
 * while the search can be performed multiple times with different approximation
 * levels.
 */
using KRAFN = RASearch<FurthestNeighborSort>;

} // namespace mlpack

#endif
