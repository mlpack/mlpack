/**
 * @file neighbor_search_rules.hpp
 * @author Ryan Curtin
 *
 * Defines the pruning rules and base case rules necessary to perform a
 * tree-based search (with an arbitrary tree) for the NeighborSearch class.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_RULES_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_RULES_HPP

namespace mlpack {
namespace neighbor {

template<typename SortPolicy, typename MetricType, typename TreeType>
class NeighborSearchRules
{
 public:
  NeighborSearchRules(const arma::mat& referenceSet,
                      const arma::mat& querySet,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances,
                      MetricType& metric);

  void BaseCase(const size_t queryIndex, const size_t referenceIndex);

  // For single-tree traversal.
  bool CanPrune(const size_t queryIndex, TreeType& referenceNode);

  // For dual-tree traversal.
  bool CanPrune(TreeType& queryNode, TreeType& referenceNode);

  // Get the order of points to recurse to.
  bool LeftFirst(const size_t queryIndex, TreeType& referenceNode);
  bool LeftFirst(TreeType& staticNode, TreeType& recurseNode);

  // Update bounds.  Needs a better name.
  void UpdateAfterRecursion(TreeType& queryNode, TreeType& referenceNode);

 private:
  //! The reference set.
  const arma::mat& referenceSet;

  //! The query set.
  const arma::mat& querySet;

  //! The matrix the resultant neighbor indices should be stored in.
  arma::Mat<size_t>& neighbors;

  //! The matrix the resultant neighbor distances should be stored in.
  arma::mat& distances;

  //! The instantiated metric.
  MetricType& metric;

  /**
   * Insert a point into the neighbors and distances matrices; this is a helper
   * function.
   *
   * @param queryIndex Index of point whose neighbors we are inserting into.
   * @param pos Position in list to insert into.
   * @param neighbor Index of reference point which is being inserted.
   * @param distance Distance from query point to reference point.
   */
  void InsertNeighbor(const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double distance);
};

}; // namespace neighbor
}; // namespace mlpack

// Include implementation.
#include "neighbor_search_rules_impl.hpp"

#endif // __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_RULES_HPP
