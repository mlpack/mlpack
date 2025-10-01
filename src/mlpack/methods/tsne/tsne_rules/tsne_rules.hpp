/**
 * @file methods/tsne/tsne_rules/tsne_rules.hpp
 * @author Ranjodh Singh
 *
 * Defines the pruning rules and base case rules necessary
 * to perform a tree-based approximation of the t-SNE Gradient Function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_HPP
#define MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/tree/hrectbound.hpp>
#include <mlpack/core/distances/lmetric.hpp>

namespace mlpack
{

/**
 * Traversal Rules class for Approximating t-SNE Gradient (Repulsive Term).
 * This class can be used by both Single and Dual Tree Traversers.
 *
 * @tparam IsDualTraversal Indicates whether the traversal is dual (true) or
           single (false). Allows both barnes-hut and dual-tree approximations
           to be handled in one class.
 * @tparam MatType The type of Matrix.
 */
template <bool IsDualTraversal, typename MatType = arma::mat>
class TSNERules
{
 public:
  // Convenience typedefs.
  using VecType = typename GetColType<MatType>::type;
  using DistanceType = SquaredEuclideanDistance;
  using HRectBoundType = HRectBound<DistanceType>;

  /**
   * Constructs TSNERules object.
   *
   * @param sumQ denominator term for the negative force.
   * @param negF nominator term for the negative force.
   * @param embedding low dimentional embedding matrix.
   * @param oldFromNew mapping form previous to new order of points.
   * @param dof Degrees of freedom calculated as max(1, input_dims - 1).
   * @param theta Determines whether to continue descending into child nodes 
   *              during traversal or to stop when the current approximation 
   *              is sufficiently accurate.
   */
  TSNERules(double& sumQ,
            MatType& negF,
            const MatType& embedding,
            const std::vector<size_t>& oldFromNew,
            const size_t dof = 1,
            const double theta = 0.5);

  /**
   * BaseCase.
   *
   * @param queryIndex Index of query point.
   * @param referenceIndex Index of reference point.
   */
  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  /**
   * Determine whether to prune the node referenceNode If so, return DBL_MAX.
   * Otherwise, return a numeric score indicating how "promising"
   * the node combination is (lower scores are better).
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   */
  template <typename TreeType>
  double Score(const size_t queryIndex, TreeType& referenceNode);

  /**
   * Check again if the referenceNode can be pruned, returning DBL_MAX if so.
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   * @param oldScore Old score produced by Score() (or Rescore()).
   */
  template <typename TreeType>
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore);

  /**
   * Determine whether to prune the node combination (queryNode, referenceNode)
   * If so, return DBL_MAX.  Otherwise, return a numeric score indicating how
   * "promising" the node combination is (lower scores are better).
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   */
  template <typename TreeType>
  double Score(TreeType& queryNode, TreeType& referenceNode);

  /**
   * Check again if the combination (queryNode, referenceNode) can be pruned,
   * returning DBL_MAX if so.
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   * @param oldScore Old score produced by Score() (or Rescore()).
   */
  template <typename TreeType>
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore);

  /**
   * Calculate the size of the largest side of the hyperrectangle given its
   * bounds.
   *
   * @param bound contains lower and higher bound of the hyperrectangle
                  in each dimention.
   */
  double getMaxSideSq(const HRectBoundType& bound) const;

  //! Defines Traversal information class for the dual-tree traversal
  class TraversalInfoType { /* Nothing To Do Here*/ };
  //! Get the traversal info.
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  //! Modify the traversal info.
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  //! Denominator term for the negative force.
  double& sumQ;
  
  //! Nominator term for the negative force.
  MatType& negF;

  //! The low dimentional embedding matrix.
  const MatType& embedding;

  //! Mapping form new to previous order of points.
  const std::vector<size_t>& oldFromNew;

  //! Degrees of freedom
  const size_t dof;

  //! The coarseness of the approximation.
  const double theta;

  //! Traversal information for the dual-tree traversal
  TraversalInfoType traversalInfo;
};

} // namespace mlpack

// Include implementation.
#include "./tsne_rules_impl.hpp"

#endif // MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_HPP
