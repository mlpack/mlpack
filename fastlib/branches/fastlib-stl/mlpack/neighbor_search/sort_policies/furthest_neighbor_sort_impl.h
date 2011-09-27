/***
 * @file nearest_neighbor_sort_impl.h
 * @author Ryan Curtin
 *
 * Implementation of templated methods for the NearestNeighborSort SortPolicy
 * class for the NeighborSearch class.
 */
#ifndef __MLPACK_NEIGHBOR_FURTHEST_NEIGHBOR_SORT_IMPL_H
#define __MLPACK_NEIGHBOR_FURTHEST_NEIGHBOR_SORT_IMPL_H

#include <mlpack/core/kernels/lmetric.h>

namespace mlpack {
namespace neighbor {

template<typename TreeType>
double FurthestNeighborSort::BestNodeToNodeDistance(TreeType* query_node,
                                                    TreeType* reference_node) {
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return query_node->bound().MaxDistance(reference_node->bound());
}

template<typename TreeType>
double FurthestNeighborSort::BestPointToNodeDistance(const arma::vec& point,
                                                     TreeType* reference_node) {
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return reference_node->bound().MaxDistance(point);
}

}; // namespace neighbor
}; // namespace mlpack

#endif
