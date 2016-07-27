/**
 * @file neighbor_search_rules_spill.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Defines the pruning rules and base case rules necessary to perform a
 * tree-based search with Spill Trees for the NeighborSearch class.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_RULES_SPILL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_RULES_SPILL_HPP

#include <mlpack/core/tree/traversal_info.hpp>
#include <mlpack/core/tree/spill_tree.hpp>

namespace mlpack {
namespace neighbor {

/**
 * NeighborSearchRules specialization for Spill Trees.
 * The main difference with the general implementation is that Score() methods
 * consider the special case of a overlapping node.
 *
 * @tparam SortPolicy The sort policy for distances.
 * @tparam MetricType The metric to use for computation.
 * @tparam TreeType The tree type to use; must adhere to the TreeType API.
 */
template<typename StatisticType,
         typename MatType,
         template<typename SplitBoundT, typename SplitMatT> class SplitType,
         typename SortPolicy,
         typename MetricType>
class NeighborSearchRules<SortPolicy, MetricType, tree::SpillTree<MetricType,
    StatisticType, MatType, SplitType>>
{
  typedef tree::SpillTree<MetricType, StatisticType, MatType, SplitType>
      TreeType;
 public:
  /**
   * Construct the NeighborSearchRules object.  This is usually done from within
   * the NeighborSearch class at search time.
   *
   * @param referenceSet Set of reference data.
   * @param querySet Set of query data.
   * @param k Number of neighbors to search for.
   * @param metric Instantiated metric.
   * @param epsilon Relative approximate error.
   * @param sameSet If true, the query and reference set are taken to be the
   *      same, and a query point will not return itself in the results.
   */
  NeighborSearchRules(const typename TreeType::Mat& referenceSet,
                      const typename TreeType::Mat& querySet,
                      const size_t k,
                      MetricType& metric,
                      const double epsilon = 0,
                      const bool sameSet = false);

  /**
   * Store the list of candidates for each query point in the given matrices.
   *
   * @param neighbors Matrix storing lists of neighbors for each query point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   */
  void GetResults(arma::Mat<size_t>& neighbors, arma::mat& distances);

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

  //! Get the number of base cases that have been performed.
  size_t BaseCases() const { return baseCases; }
  //! Modify the number of base cases that have been performed.
  size_t& BaseCases() { return baseCases; }

  //! Get the number of scores that have been performed.
  size_t Scores() const { return scores; }
  //! Modify the number of scores that have been performed.
  size_t& Scores() { return scores; }

  //! Convenience typedef.
  typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;

  //! Get the traversal info.
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  //! Modify the traversal info.
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 protected:
  //! The reference set.
  const typename TreeType::Mat& referenceSet;

  //! The query set.
  const typename TreeType::Mat& querySet;

  //! Candidate represents a possible candidate neighbor (distance, index).
  typedef std::pair<double, size_t> Candidate;

  //! Compare two candidates based on the distance.
  struct CandidateCmp {
    bool operator()(const Candidate& c1, const Candidate& c2)
    {
      return !SortPolicy::IsBetter(c2.first, c1.first);
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

  //! Denotes whether or not the reference and query sets are the same.
  bool sameSet;

  //! Relative error to be considered in approximate search.
  const double epsilon;

  //! The last query point BaseCase() was called with.
  size_t lastQueryIndex;
  //! The last reference point BaseCase() was called with.
  size_t lastReferenceIndex;
  //! The last base case result.
  double lastBaseCase;

  //! The number of base cases that have been performed.
  size_t baseCases;
  //! The number of scores that have been performed.
  size_t scores;

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
};

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "neighbor_search_rules_spill_impl.hpp"

#endif // MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_RULES_SPILL_HPP
