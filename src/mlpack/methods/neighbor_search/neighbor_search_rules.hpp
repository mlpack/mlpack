/**
 * @file neighbor_search_rules.hpp
 * @author Ryan Curtin
 *
 * Defines the pruning rules and base case rules necessary to perform a
 * tree-based search (with an arbitrary tree) for the NeighborSearch class.
 * This file is part of MLPACK 1.0.2.
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

  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  // Update bounds.  Needs a better name.
  void UpdateAfterRecursion(TreeType& queryNode, TreeType& referenceNode);

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   */
  double Score(const size_t queryIndex, TreeType& referenceNode) const;

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
               const double baseCaseResult) const;

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
  double Score(TreeType& queryNode, TreeType& referenceNode) const;

  /**
   * Get the score for recursion order, passing the base case result (in the
   * situation where it may be needed to calculate the recursion order).  A low
   * score indicates priority for recursion, while DBL_MAX indicates that the
   * node should not be recursed into at all (it should be pruned).
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   * @param baseCaseResult Result of BaseCase(queryIndex, referenceNode).
   */
  double Score(TreeType& queryNode,
               TreeType& referenceNode,
               const double baseCaseResult) const;

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
