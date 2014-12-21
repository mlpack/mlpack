/**
 * @file unmap.hpp
 * @author Ryan Curtin
 *
 * Convenience methods to unmap results.
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
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_UNMAP_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_UNMAP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace neighbor {

/**
 * Assuming that the datasets have been mapped using the referenceMap and the
 * queryMap (such as during kd-tree construction), unmap the columns of the
 * distances and neighbors matrices into neighborsOut and distancesOut, and also
 * unmap the entries in each row of neighbors.  This is useful for the dual-tree
 * case.
 *
 * @param neighbors Matrix of neighbors resulting from neighbor search.
 * @param distances Matrix of distances resulting from neighbor search.
 * @param referenceMap Mapping of reference set to old points.
 * @param queryMap Mapping of query set to old points.
 * @param neighborsOut Matrix to store unmapped neighbors into.
 * @param distancesOut Matrix to store unmapped distances into.
 * @param squareRoot If true, take the square root of the distances.
 */
void Unmap(const arma::Mat<size_t>& neighbors,
           const arma::mat& distances,
           const std::vector<size_t>& referenceMap,
           const std::vector<size_t>& queryMap,
           arma::Mat<size_t>& neighborsOut,
           arma::mat& distancesOut,
           const bool squareRoot = false);

/**
 * Assuming that the datasets have been mapped using referenceMap (such as
 * during kd-tree construction), unmap the columns of the distances and
 * neighbors matrices into neighborsOut and distancesOut, and also unmap the
 * entries in each row of neighbors.  This is useful for the single-tree case.
 *
 * @param neighbors Matrix of neighbors resulting from neighbor search.
 * @param distances Matrix of distances resulting from neighbor search.
 * @param referenceMap Mapping of reference set to old points.
 * @param neighborsOut Matrix to store unmapped neighbors into.
 * @param distancesOut Matrix to store unmapped distances into.
 * @param squareRoot If true, take the square root of the distances.
 */
void Unmap(const arma::Mat<size_t>& neighbors,
           const arma::mat& distances,
           const std::vector<size_t>& referenceMap,
           arma::Mat<size_t>& neighborsOut,
           arma::mat& distancesOut,
           const bool squareRoot = false);

}; // namespace neighbor
}; // namespace mlpack

#endif
