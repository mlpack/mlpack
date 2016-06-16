/**
 * @file ra_typedef.hpp
 * @author Parikshit Ram
 *
 * Simple typedefs describing template instantiations of the RASearch
 * class which are commonly used.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
 * The KRANN class is the k-rank-approximate-nearest-neighbors method.  It
 * returns L2 distances for each of the k rank-approximate nearest-neighbors.
 *
 * The approximation is controlled with two parameters (see allkrann_main.cpp)
 * which can be specified at search time. So the tree building is done only once
 * while the search can be performed multiple times with different approximation
 * levels.
 */
typedef RASearch<> KRANN;

/**
 * The KRAFN class is the k-rank-approximate-farthest-neighbors method.  It
 * returns L2 distances for each of the k rank-approximate farthest-neighbors.
 *
 * The approximation is controlled with two parameters (see allkrann_main.cpp)
 * which can be specified at search time. So the tree building is done only once
 * while the search can be performed multiple times with different approximation
 * levels.
 */
typedef RASearch<FurthestNeighborSort> KRAFN;

/**
 * @deprecated
 * The AllkRANN class is the all-k-rank-approximate-nearest-neighbors method.  It
 * returns L2 distances for each of the k rank-approximate nearest-neighbors.
 *
 * The approximation is controlled with two parameters (see allkrann_main.cpp)
 * which can be specified at search time. So the tree building is done only once
 * while the search can be performed multiple times with different approximation
 * levels.
 *
 * This typedef will be removed in mlpack 3.0.0; use the KRANN typedef instead.
 */
typedef RASearch<> AllkRANN;

/**
 * @deprecated
 * The AllkRAFN class is the all-k-rank-approximate-farthest-neighbors method.
 * It returns L2 distances for each of the k rank-approximate
 * farthest-neighbors.
 *
 * The approximation is controlled with two parameters (see allkrann_main.cpp)
 * which can be specified at search time. So the tree building is done only once
 * while the search can be performed multiple times with different approximation
 * levels.
 *
 * This typedef will be removed in mlpack 3.0.0; use the KRANN typedef instead.
 */
typedef RASearch<> AllkRAFN;

} // namespace neighbor
} // namespace mlpack

#endif
