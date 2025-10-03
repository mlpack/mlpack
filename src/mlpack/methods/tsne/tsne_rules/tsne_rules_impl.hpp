/**
 * @file methods/tsne/tsne_rules/tsne_rules_impl.hpp
 * @author Ranjodh Singh
 *
 * Implements the pruning rules and base case rules necessary
 * to perform a tree-based approximation of the t-SNE Gradient Function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_IMPL_HPP
#define MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_IMPL_HPP

#include "./tsne_rules.hpp"

namespace mlpack
{

template <bool IsDualTraversal, typename MatType>
TSNERules<IsDualTraversal, MatType>::TSNERules(
    double& sumQ,
    MatType& negF,
    const MatType& embedding,
    const std::vector<size_t>& oldFromNew,
    const size_t dof,
    const double theta)
    : sumQ(sumQ), negF(negF), embedding(embedding), 
    oldFromNew(oldFromNew), dof(dof), theta(theta)
{
  /* Nothing To Do Here */
}

template <bool IsDualTraversal, typename MatType>
double TSNERules<IsDualTraversal, MatType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  const VecType& queryPoint = embedding.col(oldFromNew[queryIndex]);
  const VecType& referencePoint = embedding.col(oldFromNew[referenceIndex]);
  const double distanceSq = DistanceType::Evaluate(queryPoint, referencePoint);

  if (distanceSq > arma::datum::eps)
  {
    double q = (double) dof / (dof + distanceSq);
    if (dof != 1)
      q = std::pow(q, (1.0 + dof) / 2.0);

    sumQ += q;
    negF.col(oldFromNew[queryIndex]) += q * q * (queryPoint - referencePoint);
    if constexpr (IsDualTraversal)
      negF.col(oldFromNew[referenceIndex]) += q * q *
                                              (referencePoint - queryPoint);
  }

  return distanceSq;
}

template <bool IsDualTraversal, typename MatType>
template <typename TreeType>
double TSNERules<IsDualTraversal, MatType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  const VecType& queryPoint = embedding.col(oldFromNew[queryIndex]);
  const VecType& referencePoint = referenceNode.Stat().Centroid();
  const double distanceSq = std::max(
      arma::datum::eps, DistanceType::Evaluate(queryPoint, referencePoint));

  const double maxSideSq = getMaxSideSq(referenceNode.Bound());
  if (maxSideSq / distanceSq < theta * theta)
  {
    double q = (double) dof / (dof + distanceSq);
    if (dof != 1)
      q = std::pow(q, (1.0 + dof) / 2.0);

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
template <bool IsDualTraversal, typename MatType>
template <typename TreeType>
double TSNERules<IsDualTraversal, MatType>::Rescore(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double oldScore)
{
  return oldScore;
}

template <bool IsDualTraversal, typename MatType>
template <typename TreeType>
double TSNERules<IsDualTraversal, MatType>::Score(
    TreeType& queryNode, TreeType& referenceNode)
{
  const VecType& queryPoint = queryNode.Stat().Centroid();
  const VecType& referencePoint = referenceNode.Stat().Centroid();
  const double distanceSq = std::max(
      arma::datum::eps, DistanceType::Evaluate(queryPoint, referencePoint));

  const double maxSideSq = std::max(getMaxSideSq(queryNode.Bound()),
                                    getMaxSideSq(referenceNode.Bound()));
  if (maxSideSq / distanceSq < theta * theta)
  {
    double q = (double) dof / (dof + distanceSq);
    if (dof != 1)
      q = std::pow(q, (1.0 + dof) / 2.0);

    sumQ += queryNode.NumDescendants() * referenceNode.NumDescendants() * q;
    for (size_t i = 0; i < queryNode.NumDescendants(); i++)
    {
      negF.col(oldFromNew[queryNode.Descendant(i)]) +=
          referenceNode.NumDescendants() * q * q *
          (queryPoint - referencePoint);
    }
    for (size_t i = 0; i < referenceNode.NumDescendants(); i++)
    {
      negF.col(oldFromNew[referenceNode.Descendant(i)]) +=
          queryNode.NumDescendants() * q * q *
          (referencePoint - queryPoint);
    }
      
    return DBL_MAX;
  }
  else
  {
    return maxSideSq / distanceSq;
  }
}

template <bool IsDualTraversal, typename MatType>
template <typename TreeType>
double TSNERules<IsDualTraversal, MatType>::Rescore(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double oldScore)
{
  return oldScore;
}

template <bool IsDualTraversal, typename MatType>
double TSNERules<IsDualTraversal, MatType>::getMaxSideSq(
    const HRectBoundType& bound) const
{
  double maxSide = 0.0;
  for (size_t i = 0; i < bound.Dim(); i++)
    maxSide = std::max(maxSide, bound[i].Hi() - bound[i].Lo());
  return maxSide * maxSide;
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_IMPL_HPP
