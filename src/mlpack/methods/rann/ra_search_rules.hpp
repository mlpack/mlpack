/**
 * @file ra_search_rules.hpp
 * @author Parikshit Ram
 *
 * Defines the pruning rules and base case rules necessary to perform a
 * tree-based rank-approximate search (with an arbitrary tree) for the RASearch
 * class.
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
#ifndef __MLPACK_METHODS_RANN_RA_SEARCH_RULES_HPP
#define __MLPACK_METHODS_RANN_RA_SEARCH_RULES_HPP

#include "../neighbor_search/ns_traversal_info.hpp"
#include "ra_search.hpp" // For friend declaration.

namespace mlpack {
namespace neighbor {

template<typename SortPolicy, typename MetricType, typename TreeType>
class RASearchRules
{
 public:
  RASearchRules(const arma::mat& referenceSet,
                const arma::mat& querySet,
                arma::Mat<size_t>& neighbors,
                arma::mat& distances,
                MetricType& metric,
                const double tau = 5,
                const double alpha = 0.95,
                const bool naive = false,
                const bool sampleAtLeaves = false,
                const bool firstLeafExact = false,
                const size_t singleSampleLimit = 20);



  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * For rank-approximation, the scoring function first checks if pruning
   * by distance is possible.
   *   If yes, then the node is given the score of
   *   'DBL_MAX' and the expected number of samples from that node are
   *   added to the number of samples made for the query.
   *
   *   If no, then the function tries to see if the node can be pruned by
   *   approximation. If number of samples required from this node is small
   *   enough, then that number of samples are acquired from this node
   *   and the score is set to be 'DBL_MAX'.
   *
   *   If the pruning by approximation is not possible either, the algorithm
   *   continues with the usual tree-traversal.
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   */
  double Score(const size_t queryIndex, TreeType& referenceNode);

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * For rank-approximation, the scoring function first checks if pruning
   * by distance is possible.
   *   If yes, then the node is given the score of
   *   'DBL_MAX' and the expected number of samples from that node are
   *   added to the number of samples made for the query.
   *
   *   If no, then the function tries to see if the node can be pruned by
   *   approximation. If number of samples required from this node is small
   *   enough, then that number of samples are acquired from this node
   *   and the score is set to be 'DBL_MAX'.
   *
   *   If the pruning by approximation is not possible either, the algorithm
   *   continues with the usual tree-traversal.
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
   * for recursion, while DBL_MAX indicates that the node should not be
   * recursed into at all (it should be pruned).  This is used when the score
   * has already been calculated, but another recursion may have modified the
   * bounds for pruning.  So the old score is checked against the new pruning
   * bound.
   *
   * For rank-approximation, it also checks if the number of samples left
   * for a query to satisfy the rank constraint is small enough at this
   * point of the algorithm, then this node is approximated by sampling
   * and given a new score of 'DBL_MAX'.
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
   * recursionm while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * For the rank-approximation, we check if the referenceNode can be
   * approximated by sampling. If it can be, enough samples are made for
   * every query in the queryNode. No further query-tree traversal is
   * performed.
   *
   * The 'NumSamplesMade' query stat is propagated up the tree. And then
   * if pruning occurs (by distance or by sampling), the 'NumSamplesMade'
   * stat is not propagated down the tree. If no pruning occurs, the
   * stat is propagated down the tree.
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
   * For the rank-approximation, we check if the referenceNode can be
   * approximated by sampling. If it can be, enough samples are made for
   * every query in the queryNode. No further query-tree traversal is
   * performed.
   *
   * The 'NumSamplesMade' query stat is propagated up the tree. And then
   * if pruning occurs (by distance or by sampling), the 'NumSamplesMade'
   * stat is not propagated down the tree. If no pruning occurs, the
   * stat is propagated down the tree.
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   * @param baseCaseResult Result of BaseCase(queryIndex, referenceNode).
   */
  double Score(TreeType& queryNode,
               TreeType& referenceNode,
               const double baseCaseResult);

  /**
   * Re-evaluate the score for recursion order.  A low score indicates priority
   * for recursion, while DBL_MAX indicates that the node should not be
   * recursed into at all (it should be pruned).  This is used when the score
   * has already been calculated, but another recursion may have modified the
   * bounds for pruning.  So the old score is checked against the new pruning
   * bound.
   *
   * For the rank-approximation, we check if the referenceNode can be
   * approximated by sampling. If it can be, enough samples are made for
   * every query in the queryNode. No further query-tree traversal is
   * performed.
   *
   * The 'NumSamplesMade' query stat is propagated up the tree. And then
   * if pruning occurs (by distance or by sampling), the 'NumSamplesMade'
   * stat is not propagated down the tree. If no pruning occurs, the
   * stat is propagated down the tree.
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   * @param oldScore Old score produced by Socre() (or Rescore()).
   */
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore);


  size_t NumDistComputations() { return numDistComputations; }
  size_t NumEffectiveSamples()
  {
    if (numSamplesMade.n_elem == 0)
      return 0;
    else
      return arma::sum(numSamplesMade);
  }

  typedef neighbor::NeighborSearchTraversalInfo<TreeType> TraversalInfoType;

  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

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

  //! Whether to sample at leaves or just use all of it
  bool sampleAtLeaves;

  //! Whether to do exact computation on the first leaf before any sampling
  bool firstLeafExact;

  //! The limit on the largest node that can be approximated by sampling
  size_t singleSampleLimit;

  //! The minimum number of samples required per query
  size_t numSamplesReqd;

  //! The number of samples made for every query
  arma::Col<size_t> numSamplesMade;

  //! The sampling ratio
  double samplingRatio;

  // TO REMOVE: just for testing
  size_t numDistComputations;

  TraversalInfoType traversalInfo;

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

  /**
   * Compute the minimum number of samples required to guarantee
   * the given rank-approximation and success probability.
   *
   * @param n Size of the set to be sampled from.
   * @param k The number of neighbors required within the rank-approximation.
   * @param tau The rank-approximation in percentile of the data.
   * @param alpha The success probability desired.
   */
  size_t MinimumSamplesReqd(const size_t n,
                            const size_t k,
                            const double tau,
                            const double alpha) const;

  /**
   * Compute the success probability of obtaining 'k'-neighbors from a
   * set of size 'n' within the top 't' neighbors if 'm' samples are made.
   *
   * @param n Size of the set being sampled from.
   * @param k The number of neighbors required within the rank-approximation.
   * @param m The number of random samples.
   * @param t The desired rank-approximation.
   */
  double SuccessProbability(const size_t n,
                            const size_t k,
                            const size_t m,
                            const size_t t) const;

  /**
   * Pick up desired number of samples (with replacement) from a given range
   * of integers so that only the distinct samples are returned from
   * the range [0 - specified upper bound)
   *
   * @param numSamples Number of random samples.
   * @param rangeUpperBound The upper bound on the range of integers.
   * @param distinctSamples The list of the distinct samples.
   */
  void ObtainDistinctSamples(const size_t numSamples,
                             const size_t rangeUpperBound,
                             arma::uvec& distinctSamples) const;

  /**
   * Perform actual scoring for single-tree case.
   */
  double Score(const size_t queryIndex,
               TreeType& referenceNode,
               const double distance,
               const double bestDistance);

  /**
   * Perform actual scoring for dual-tree case.
   */
  double Score(TreeType& queryNode,
               TreeType& referenceNode,
               const double distance,
               const double bestDistance);

  // So that RASearch can access ObtainDistinctSamples() and
  // MinimumSamplesReqd().  Maybe refactoring is a better solution but this is
  // okay for now.
  friend class RASearch<SortPolicy, MetricType, TreeType>;
}; // class RASearchRules

}; // namespace neighbor
}; // namespace mlpack

// Include implementation.
#include "ra_search_rules_impl.hpp"

#endif // __MLPACK_METHODS_RANN_RA_SEARCH_RULES_HPP
