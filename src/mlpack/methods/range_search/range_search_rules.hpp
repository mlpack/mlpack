/**
 * @file methods/range_search/range_search_rules.hpp
 * @author Ryan Curtin
 *
 * Rules for range search, so that it can be done with arbitrary tree types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_RULES_HPP

#include <mlpack/core/tree/traversal_info.hpp>

namespace mlpack {

/**
 * The RangeSearchRules class is a template helper class used by RangeSearch
 * class when performing range searches.
 *
 * @tparam DistanceType The distance metric to use for computation.
 * @tparam TreeType The tree type to use; must adhere to the TreeType API.
 */
template<typename DistanceType, typename TreeType>
class RangeSearchRules
{
 public:
  //! Easy access to MatType.
  using MatType = typename TreeType::Mat;
  //! The type of element held in MatType.
  using ElemType = typename MatType::elem_type;

  /**
   * Construct the RangeSearchRules object.  This is usually done from within
   * the RangeSearch class at search time.
   *
   * @param referenceSet Set of reference data.
   * @param querySet Set of query data.
   * @param range Range to search for.
   * @param neighbors Vector to store resulting neighbors in.
   * @param distances Vector to store resulting distances in.
   * @param distance Instantiated distance metric.
   * @param sameSet If true, the query and reference set are taken to be the
   *      same, and a query point will not return itself in the results.
   */
  RangeSearchRules(const MatType& referenceSet,
                   const MatType& querySet,
                   const RangeType<ElemType>& range,
                   std::vector<std::vector<size_t> >& neighbors,
                   std::vector<std::vector<ElemType> >& distances,
                   DistanceType& distance,
                   const bool sameSet = false);

  /**
   * Compute the base case between the given query point and reference point.
   *
   * @param queryIndex Index of query point.
   * @param referenceIndex Index of reference point.
   */
  ElemType BaseCase(const size_t queryIndex, const size_t referenceIndex);

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   */
  ElemType Score(const size_t queryIndex, TreeType& referenceNode);

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
  ElemType Rescore(const size_t queryIndex,
                   TreeType& referenceNode,
                   const ElemType oldScore) const;

  /**
   * Get the score for recursion order.  A low score indicates priority for
   * recursion, while DBL_MAX indicates that the node should not be recursed
   * into at all (it should be pruned).
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   */
  ElemType Score(TreeType& queryNode, TreeType& referenceNode);

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
  ElemType Rescore(TreeType& queryNode,
                   TreeType& referenceNode,
                   const ElemType oldScore) const;

  using TraversalInfoType = mlpack::TraversalInfo<TreeType>;

  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

  //! Get the number of base cases.
  size_t BaseCases() const { return baseCases; }
  //! Get the number of scores (that is, calls to RangeDistance()).
  size_t Scores() const { return scores; }

  //! Get the minimum number of base cases we need to perform to have acceptable
  //! results.
  size_t MinimumBaseCases() const { return 0; }

 private:
  //! The reference set.
  const MatType& referenceSet;

  //! The query set.
  const MatType& querySet;

  //! The range of distances for which we are searching.
  const RangeType<ElemType>& range;

  //! The vector the resultant neighbor indices should be stored in.
  std::vector<std::vector<size_t> >& neighbors;

  //! The vector the resultant neighbor distances should be stored in.
  std::vector<std::vector<ElemType> >& distances;

  //! The instantiated distance metric.
  DistanceType& distance;

  //! If true, the query and reference set are taken to be the same.
  bool sameSet;

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

  //! The number of base cases.
  size_t baseCases;
  //! THe number of scores.
  size_t scores;
};

} // namespace mlpack

// Include implementation.
#include "range_search_rules_impl.hpp"

#endif
