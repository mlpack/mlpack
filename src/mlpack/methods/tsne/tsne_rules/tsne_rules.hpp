/**
 * @file methods/tsne/tsne_rules/tsne_rules.hpp
 * @author Ranjodh Singh
 *
 * Defines the pruning rules and base case rules required
 * to perform a tree-based approximation of the t-SNE gradient.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_HPP
#define MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/tree/octree.hpp>

#include "./tsne_centroid_statistic.hpp"

namespace mlpack {

/**
 * Traversal rules for approximating the repulsive part of the t-SNE gradient.
 *
 * This class supports both single and dual-tree traversers:
 * - With a single-tree traverser, it performs the Barnes-Hut approximation.
 * - With a dual-tree traverser, it performs the dual-tree approximation.
 *
 * See "Accelerating t-SNE using Tree-Based Algorithms" for more details.
 *
 * @tparam MatType The type of Matrix.
 */
template <typename MatType>
class TSNERules
{
 public:
  // Convenience typedefs.
  using ElemType = typename MatType::elem_type;
  using VecType = typename GetColType<MatType>::type;
  using DistanceType = SquaredEuclideanDistance;

  /**
   * Constructs the TSNERules object.
   *
   * @param sumQ Normalization value for the repulsive forces.
   * @param repF Matrix used to store the repulsive forces.
   * @param embedding Low-dimentional embedding matrix.
   * @param oldFromNew Mapping from the previous to the new order of points.
   * @param dof Degrees of freedom.
   * @param theta Coarseness of the approximation.
   */
  TSNERules(double& sumQ,
            MatType& repF,
            const MatType& embedding,
            const std::vector<size_t>& oldFromNew,
            const size_t dof,
            const double theta);

  /**
   * Computes point-point interactions for the repulsive term.
   *
   * @param queryIndex Index of query point.
   * @param referenceIndex Index of reference point.
   */
  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  /**
   * Computes point-to-node interactions for the repulsive term.
   *
   * Determines whether to prune the reference node using the Barnes-Hut
   * approximation criterion:
   * \f[
   * \frac{\text{diameter}}{\text{distance}} < \theta
   * \f]
   * where `theta` controls the trade-off between computational speed and
   * accuracy.
   *
   * If the condition is satisfied, the reference node is pruned
   * (`DBL_MAX` is returned to indicate the same), and the repulsive force
   * contribution is computed immediately using the distance between the
   * query point and the node centroid as an approximation for the distances
   * between the query point and all points within the reference node.
   * Otherwise, the computed ratio (`diameter / distance`) is returned,
   * indicating that the traverser should descend further into the
   * reference node.
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Candidate node to be recursed into.
   */
  template <typename TreeType>
  double Score(const size_t queryIndex, TreeType& referenceNode);

  /**
   * Check again if the referenceNode can be pruned, returning `DBL_MAX` if so.
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
   * Computes node-to-node interactions for the repulsive term.
   *
   * Determines whether to prune the combination (queryNode, referenceNode)
   * using the dual-tree approximation criterion:
   * \f[
   * \frac{\max(\text{queryNodeDiameter}, \text{referenceNodeDiameter})}
   * {\text{distance}} < \theta
   * \f]
   * where `theta` controls the trade-off between computational speed and
   * accuracy.
   *
   * If the condition is satisfied, the combination (queryNode, referenceNode)
   * is pruned (`DBL_MAX` is returned to indicate the same), and the repulsive
   * force contributions are computed immediately using the distance between
   * the queryNode centroid and the referenceNode centroid as an approximation
   * for the distances between all point pairs, where one point belongs to the
   * queryNode and the other to the referenceNode.
   * Otherwise, the computed ratio
   * (`max(queryNodeDiameter, referenceNodeDiameter) / distance`)
   * is returned, indicating that the traverser should descend further into
   * the node combination (queryNode, referenceNode).
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   */
  template <typename TreeType>
  double Score(TreeType& queryNode, TreeType& referenceNode);

  /**
   * Check again if the combination (queryNode, referenceNode) can be pruned,
   * returning `DBL_MAX` if so.
   *
   * @param queryNode Candidate query node to recurse into.
   * @param referenceNode Candidate reference node to recurse into.
   * @param oldScore Old score produced by Score() (or Rescore()).
   */
  template <typename TreeType>
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore);

  //! Traversal information class for the dual-tree traversals
  class TraversalInfoType { /* Nothing To Do Here */ };

  //! Get the traversal info.
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  //! Modify the traversal info.
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  //! Normalization term for the repulsive forces.
  double& sumQ;

  //! Matrix used to store the repulsive forces.
  MatType& repF;

  //! Low-dimentional embedding matrix.
  const MatType& embedding;

  //! Mapping from new to previous order of points.
  const std::vector<size_t>& oldFromNew;

  //! Degrees of freedom
  const size_t dof;

  //! Coarseness of the approximation.
  const double theta;

  //! Traversal information for the dual-tree traversal
  TraversalInfoType traversalInfo;
};

} // namespace mlpack

// Include implementation.
#include "./tsne_rules_impl.hpp"

#endif // MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_HPP
