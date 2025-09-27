/**
 * @file methods/tsne/tsne_approx_rules.hpp
 * @author Ranjodh Singh
 *
 * Defines and Implements the pruning rules and base case rules necessary
 * to perform a tree-based approximation of the t-SNE Gradient Function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_APPROX_RULES_HPP
#define MLPACK_METHODS_TSNE_TSNE_APPROX_RULES_HPP

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
class TSNEApproxRules
{
 public:
  // Convenience typedefs.
  using VecType = typename GetColType<MatType>::type;
  using DistanceType = SquaredEuclideanDistance;
  using HRectBoundType = HRectBound<DistanceType>;

  TSNEApproxRules(double& sumQ,
                  MatType& negF,
                  const MatType& embedding,
                  const std::vector<size_t>& oldFromNew,
                  const double theta = 0.5)
      : sumQ(sumQ), negF(negF), embedding(embedding), oldFromNew(oldFromNew),
        theta(theta)
  {
    // Nothing To Do Here
  }

  double BaseCase(const size_t queryIndex, const size_t referenceIndex)
  {
    if (queryIndex == referenceIndex)
      return 0.0;

    const VecType& queryPoint = embedding.col(oldFromNew[queryIndex]);
    const VecType& referencePoint = embedding.col(oldFromNew[referenceIndex]);
    const double distanceSq = DistanceType::Evaluate(queryPoint,
                                                     referencePoint);

    if (distanceSq > arma::datum::eps)
    {
      const double q = 1.0 / (1.0 + distanceSq);

      sumQ += q;
      negF.col(oldFromNew[queryIndex]) += q * q *
                                          (queryPoint - referencePoint);
      if constexpr (IsDualTraversal)
        negF.col(oldFromNew[referenceIndex]) += q * q *
                                                (referencePoint - queryPoint);
    }

    return distanceSq;
  }

  template <typename TreeType>
  double Score(const size_t queryIndex, TreeType& referenceNode)
  {
    const VecType& queryPoint = embedding.col(oldFromNew[queryIndex]);
    const VecType& referencePoint = referenceNode.Stat().Centroid();
    const double distanceSq = std::max(
        arma::datum::eps, DistanceType::Evaluate(queryPoint, referencePoint));

    const double maxSideSq = getMaxSideSq(referenceNode.Bound());
    if (maxSideSq / distanceSq < theta * theta)
    {
      const double q = 1.0 / (1.0 + distanceSq);

      sumQ += referenceNode.NumDescendants() * q;
      negF.col(oldFromNew[queryIndex]) += referenceNode.NumDescendants() * q *
                                          q * (queryPoint - referencePoint);
      return DBL_MAX;
    }
    else
    {
      return maxSideSq / distanceSq;
    }
  }

  template <typename TreeType>
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore)
  {
    return oldScore;
  }

  template <typename TreeType>
  double Score(TreeType& queryNode, TreeType& referenceNode)
  {
    const VecType& queryPoint = queryNode.Stat().Centroid();
    const VecType& referencePoint = referenceNode.Stat().Centroid();
    const double distanceSq = std::max(
        arma::datum::eps, DistanceType::Evaluate(queryPoint, referencePoint));

    const double maxSideSq = std::max(getMaxSideSq(queryNode.Bound()),
                                      getMaxSideSq(referenceNode.Bound()));
    if (maxSideSq / distanceSq < theta * theta)
    {
      const double q = 1.0 / (1.0 + distanceSq);

      sumQ += queryNode.NumDescendants() * referenceNode.NumDescendants() * q;
      for (size_t i = 0; i < queryNode.NumDescendants(); i++)
        negF.col(oldFromNew[queryNode.Descendant(
            i)]) += referenceNode.NumDescendants() * q * q *
                    (queryPoint - referencePoint);
      for (size_t i = 0; i < referenceNode.NumDescendants(); i++)
        negF.col(oldFromNew[referenceNode.Descendant(
            i)]) += queryNode.NumDescendants() * q * q *
                    (referencePoint - queryPoint);
      return DBL_MAX;
    }
    else
    {
      return maxSideSq / distanceSq;
    }
  }

  template <typename TreeType>
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore)
  {
    return oldScore;
  }

  double getMaxSideSq(const HRectBoundType& bound) const
  {
    double maxSide = 0.0;
    for (size_t i = 0; i < bound.Dim(); i++)
      maxSide = std::max(maxSide, bound[i].Hi() - bound[i].Lo());
    return maxSide * maxSide;
  }

  class TraversalInfoType
  {
    /* Nothing To Do Here */
  };
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  double& sumQ;
  MatType& negF;
  const MatType& embedding;
  const std::vector<size_t>& oldFromNew;
  const double theta;

  TraversalInfoType traversalInfo;
};

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_APPROX_RULES_HPP
