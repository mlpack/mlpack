/**
 * @file lmnn_targets_rules.hpp
 * @author Ryan Curtin
 *
 * Defines the pruning rules and base case rules necessary to perform a
 * tree-based search (with an arbitrary tree) for impostors and true neighbors
 * simultaneously for the LMNN class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_LMNN_TARGETS_RULES_HPP
#define MLPACK_METHODS_LMNN_LMNN_TARGETS_RULES_HPP

#include <mlpack/core/tree/traversal_info.hpp>

#include <queue>

namespace mlpack {
namespace lmnn {

/**
 * The LMNNTargetsRules class is a template helper class used by
 * the LMNN Constraints class when performing distance-based neighbor searches
 * for points of different classes.  It is very closely related to
 * NeighborSearchRules, but the problem is somewhat simplified.  Here we want to
 * find the points in the same set that are neighbors.
 *
 * @tparam MetricType The metric to use for computation.
 * @tparam TreeType The tree type to use; must adhere to the TreeType API.
 */
template<typename MetricType, typename TreeType>
class LMNNTargetsRules
{
 public:
  /**
   * Construct the LMNNTargetsAndImpostorsRules object.  This is usually done
   * from within the Constraints class at search time.
   *
   * @param dataset Set of reference data.
   * @param k Number of neighbors to search for.
   * @param metric Instantiated metric.
   */
  LMNNTargetsRules(const typename TreeType::Mat& dataset,
                   const size_t k,
                   MetricType& metric);

  /**
   * Store the list of candidates for each query point in the given matrices.
   *
   * @param oldFromNew Index transformations for each point.
   * @param neighbors Matrix storing lists of neighbors for each query point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   */
  void GetResults(const std::vector<size_t>& oldFromNew,
                  arma::Mat<size_t>& neighbors,
                  arma::mat& neighborDistances);

  /**
   * Get the distance from the query point to the reference point.
   * This will update the list of candidates with the new point if appropriate
   * and will track the number of base cases (number of points evaluated).
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
                 const double oldScore) const;

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursionm while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   */
  double Score(TreeType& queryNode, TreeType& referenceNode);

  /**
   * Re-evaluate the score for recursion order.  A low score indicates priority
   * for recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).  This is used when the score has already
   * been calculated, but another recursion may have modified the bounds for
   * pruning.  So the old score is checked against the new pruning bound.
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   * @param oldScore Old score produced by Socre() (or Rescore()).
   */
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore) const;

  //! Convenience typedef.
  typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;

  //! Get the traversal info.
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  //! Modify the traversal info.
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 protected:
  //! The reference set.
  const typename TreeType::Mat& dataset;

  //! Candidate represents a possible candidate neighbor (distance, index).
  typedef std::pair<double, size_t> Candidate;

  //! Compare two candidates based on the distance.
  struct CandidateCmp {
    bool operator()(const Candidate& c1, const Candidate& c2)
    {
      return !(c2.first <= c1.first);
    };
  };

  //! Use a priority queue to represent the list of candidate neighbors.
  typedef std::priority_queue<Candidate, std::vector<Candidate>, CandidateCmp>
      CandidateList;

  //! Set of candidate neighbors for each point.
  std::vector<CandidateList> candidates;

  //! Number of neighbors to search for.
  const size_t k;

  //! The instantiated metric.
  MetricType& metric;

  //! The last query point BaseCase() was called with.
  size_t lastQueryIndex;
  //! The last reference point BaseCase() was called with.
  size_t lastReferenceIndex;
  //! The last base case result.
  double lastBaseCase;

  //! Traversal info for the parent combination; this is updated by the
  //! traversal before each call to Score().
  TraversalInfoType traversalInfo;

  /**
   * Recalculate the bound for a given query node.
   */
  double CalculateBound(TreeType& queryNode) const;

  /**
   * Helper function to insert a point into the list of candidate points.
   *
   * @param queryIndex Index of point whose neighbors we are inserting into.
   * @param neighbor Index of reference point which is being inserted.
   * @param distance Distance from query point to reference point.
   */
  void InsertNeighbor(const size_t queryIndex,
                      const size_t neighbor,
                      const double distance);

  /**
   * Helper function to insert a point into the list of candidate impostors.
   *
   * @param queryIndex Index of point whose impostors we are inserting into.
   * @param neighbor Index of reference point which is being inserted.
   * @param distance Distance from query point to reference point.
   */
  void InsertImpostor(const size_t queryIndex,
                      const size_t neighbor,
                      const double distance);
};

} // namespace lmnn
} // namespace mlpack

// Include implementation.
#include "lmnn_targets_rules_impl.hpp"

#endif // MLPACK_METHODS_LMNN_LMNN_TARGETS_RULES_HPP
