/**
 * @file methods/emst/dtb_rules.hpp
 * @author Bill March (march@gatech.edu)
 *
 * Tree traverser rules for the DualTreeBoruvka algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_EMST_DTB_RULES_HPP
#define MLPACK_METHODS_EMST_DTB_RULES_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/core/tree/traversal_info.hpp>

namespace mlpack {

template<typename DistanceType, typename TreeType>
class DTBRules
{
 public:
  DTBRules(const arma::mat& dataSet,
           UnionFind& connections,
           arma::vec& neighborsDistances,
           arma::Col<size_t>& neighborsInComponent,
           arma::Col<size_t>& neighborsOutComponent,
           DistanceType& distance);

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
                 const double oldScore);

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

  using TraversalInfoType = mlpack::TraversalInfo<TreeType>;

  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

  //! Get the number of base cases performed.
  size_t BaseCases() const { return baseCases; }
  //! Modify the number of base cases performed.
  size_t& BaseCases() { return baseCases; }

  //! Get the number of node combinations that have been scored.
  size_t Scores() const { return scores; }
  //! Modify the number of node combinations that have been scored.
  size_t& Scores() { return scores; }

 private:
  //! The data points.
  const arma::mat& dataSet;

  //! Stores the tree structure so far
  UnionFind& connections;

  //! The distance to the candidate nearest neighbor for each component.
  arma::vec& neighborsDistances;

  //! The index of the point in the component that is an endpoint of the
  //! candidate edge.
  arma::Col<size_t>& neighborsInComponent;

  //! The index of the point outside of the component that is an endpoint
  //! of the candidate edge.
  arma::Col<size_t>& neighborsOutComponent;

  //! The instantiated distance metric.
  DistanceType& distance;

  /**
   * Update the bound for the given query node.
   */
  inline double CalculateBound(TreeType& queryNode) const;

  TraversalInfoType traversalInfo;

  //! The number of base cases calculated.
  size_t baseCases;
  //! The number of node combinations that have been scored.
  size_t scores;
}; // class DTBRules

} // namespace mlpack

#include "dtb_rules_impl.hpp"

#endif
