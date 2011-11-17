/***
 * @file nearest_neighbor_sort_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated methods for the NearestNeighborSort SortPolicy
 * class for the NeighborSearch class.
 */
#ifndef __MLPACK_NEIGHBOR_NEAREST_NEIGHBOR_SORT_IMPL_HPP
#define __MLPACK_NEIGHBOR_NEAREST_NEIGHBOR_SORT_IMPL_HPP

#include <mlpack/core/kernels/lmetric.hpp>

namespace mlpack {
namespace neighbor {

template<typename TreeType>
double NearestNeighborSort::BestNodeToNodeDistance(
    const TreeType* query_node,
    const TreeType* reference_node) {
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return query_node->bound().MinDistance(reference_node->bound());
}

template<typename elem_type, typename TreeType>
double NearestNeighborSort::BestPointToNodeDistance(
    const arma::Col<elem_type>& point,
    const TreeType* reference_node) {
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return reference_node->bound().MinDistance(point);
}

}; // namespace neighbor
}; // namespace mlpack

#endif
