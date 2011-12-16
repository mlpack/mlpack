/***
 * @file furthest_neighbor_sort_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated methods for the FurthestNeighborSort SortPolicy
 * class for the NeighborSearch class.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_FURTHEST_NEIGHBOR_SORT_IMPL_HPP

namespace mlpack {
namespace neighbor {

template<typename TreeType>
double FurthestNeighborSort::BestNodeToNodeDistance(
    const TreeType* queryNode,
    const TreeType* referenceNode)
{
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return queryNode->Bound().MaxDistance(referenceNode->Bound());
}

template<typename TreeType>
double FurthestNeighborSort::BestPointToNodeDistance(
    const arma::vec& point,
    const TreeType* referenceNode)
{
  // This is not implemented yet for the general case because the trees do not
  // accept arbitrary distance metrics.
  return referenceNode->Bound().MaxDistance(point);
}

}; // namespace neighbor
}; // namespace mlpack

#endif
