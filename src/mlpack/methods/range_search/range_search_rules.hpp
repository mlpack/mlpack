/**
 * @file range_search_rules.hpp
 * @author Ryan Curtin
 *
 * Rules for range search, so that it can be done with arbitrary tree types.
 */
#ifndef __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_HPP
#define __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_HPP

namespace mlpack {
namespace neighbor {

template<typename MetricType, typename TreeType>
class RangeSearchRules
{
 public:
  RangeSearchRules(const arma::mat& referenceSet,
                   const arma::mat& querySet,
                   std::vector<std::vector<size_t> >& neighbors,
                   std::vector<std::vector<double> >& distances,
                   math::Range& range,
                   MetricType& metric);

  /**
   * Compute the base case between the given query point and reference point.
   *
   * @param queryIndex Index of query point.
   * @param referenceIndex Index of reference point.
   */
  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   */
  double Score(const size_t queryIndex, TreeType& referenceNode);

  /**
   * Get the score for recursion order, passing the base case result (in the
   * situation where it may be needed to calculate the recursion order).  A low
   * score indicates priority for recursion, while DBL_MAX indicates that the
   * node should not be recursed into at all (it should be pruned).
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   * @param baseCaseResult Result of BaseCase(queryIndex, referenceNode).
   */
  double Score(const size_t queryIndex,
               TreeType& referenceNode,
               const double baseCaseResult);

  /**
   * Re-evaluate the score for recursion order.  A low score indicates priority
   * for recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).  This is used when the score has already
   * been calculated, but another recursion may have modified the bounds for
   * pruning.  So the old score is checked against the new pruning bound.
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   * @param oldScore Old score produced by Score() (or Rescore()).
   */
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore);

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   */
  double Score(TreeType& queryNode, TreeType& referenceNode);

  /**
   * Get the score for recursion order, passing the base case result (in the
   * situation where it may be needed to calculate the recursion order).  A low
   * score indicates priority for recursion, while DBL_MAX indicates that the
   * node should not be recursed into at all (it should be pruned).
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   * @param baseCaseResult Result of BaseCase(queryNode, referenceNode).
   */
  double Score(TreeType& queryNode,
               TreeType& referenceNode,
               const double baseCaseResult);

  /**
   * Re-evaluate the score for recursion order.  A low score indicates priority
   * for recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).  This is used when the score has already
   * been calculated, but another recursion may have modified the bounds for
   * pruning.  So the old score is checked against the new pruning bound.
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   * @param oldScore Old score produced by Score() (or Rescore()).
   */
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore);

 private:
  //! The reference set.
  const arma::mat& referenceSet;

  //! The query set.
  const arma::mat& querySet;

  //! The vector the resultant neighbor indices should be stored in.
  std::vector<std::vector<size_t> >& neighbors;

  //! The vector the resultant neighbor distances should be stored in.
  std::vector<std::vector<double> >& distances;

  //! The range of distances for which we are searching.
  math::Range& range;

  //! The instantiated metric.
  MetricType& metric;

  //! Add all the points in the given node to the results for the given query
  //! point.
  void AddResult(const size_t queryIndex, TreeType& referenceNode);
};

}; // namespace neighbor
}; // namespace mlpack

// Include implementation.
#include "range_search_rules_impl.hpp"

#endif
