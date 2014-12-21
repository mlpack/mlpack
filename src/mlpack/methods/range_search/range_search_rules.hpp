/**
 * @file range_search_rules.hpp
 * @author Ryan Curtin
 *
 * Rules for range search, so that it can be done with arbitrary tree types.
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_HPP
#define __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_HPP

#include "../neighbor_search/ns_traversal_info.hpp"

namespace mlpack {
namespace range {


template<typename MetricType, typename TreeType>
class RangeSearchRules
{
 public:
  /**
   * Construct the RangeSearchRules object.  This is usually done from within
   * the RangeSearch class at search time.
   *
   * @param referenceSet Set of reference data.
   * @param querySet Set of query data.
   * @param range Range to search for.
   * @param neighbors Vector to store resulting neighbors in.
   * @param distances Vector to store resulting distances in.
   * @param metric Instantiated metric.
   */
  RangeSearchRules(const arma::mat& referenceSet,
                   const arma::mat& querySet,
                   const math::Range& range,
                   std::vector<std::vector<size_t> >& neighbors,
                   std::vector<std::vector<double> >& distances,
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
   * recursion, while DBL_MAX indicates that the node should not be recursed
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
   * @param oldScore Old score produced by Score() (or Rescore()).
   */
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore) const;

  typedef neighbor::NeighborSearchTraversalInfo<TreeType> TraversalInfoType;

  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  //! The reference set.
  const arma::mat& referenceSet;

  //! The query set.
  const arma::mat& querySet;

  //! The range of distances for which we are searching.
  const math::Range& range;

  //! The vector the resultant neighbor indices should be stored in.
  std::vector<std::vector<size_t> >& neighbors;

  //! The vector the resultant neighbor distances should be stored in.
  std::vector<std::vector<double> >& distances;

  //! The instantiated metric.
  MetricType& metric;

  //! The last query index.
  size_t lastQueryIndex;
  //! The last reference index.
  size_t lastReferenceIndex;

  //! Add all the points in the given node to the results for the given query
  //! point.  If the base case has already been calculated, we make sure to not
  //! add that to the results twice.
  void AddResult(const size_t queryIndex,
                 TreeType& referenceNode);

  TraversalInfoType traversalInfo;
};

}; // namespace range
}; // namespace mlpack

// Include implementation.
#include "range_search_rules_impl.hpp"

#endif
