/**
 * @file neighbor_search_cover_rules.hpp
 * @author Ryan Curtin
 *
 * NeighborSearchCoverRules - rules for the search for neighbors using a cover
 * tree.  This is a mess.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_COVER_RULES_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_COVER_RULES_HPP

namespace mlpack {
namespace neighbor {

template<typename SortPolicy, typename MetricType, typename TreeType>
class NeighborSearchCoverRules
{
 public:
  NeighborSearchCoverRules(const arma::mat& referenceSet,
                           const arma::mat& querySet,
                           arma::Mat<size_t>& neighbors,
                           arma::mat& distances,
                           MetricType& metric);

  void BaseCase(const size_t queryIndex, const size_t referenceIndex);

  bool CanPrune(TreeType& queryNode, TreeType& referenceNode);

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
#include "neighbor_search_cover_rules_impl.hpp"

#endif
