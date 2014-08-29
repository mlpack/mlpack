/**
 * @file ra_typedef.hpp
 * @author Parikshit Ram
 *
 * Simple typedefs describing template instantiations of the RASearch
 * class which are commonly used.
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
