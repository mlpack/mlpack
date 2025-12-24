/**
 * @file methods/tsne/tsne_rules/tsne_rules_impl.hpp
 * @author Ranjodh Singh
 *
 * Implements the pruning rules and base case rules required
 * to perform a tree-based approximation of the t-SNE gradient.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_IMPL_HPP
#define MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_IMPL_HPP

#include "./tsne_rules.hpp"

namespace mlpack {

template <typename MatType>
TSNERules<MatType>::TSNERules(
    double& sumQ,
    MatType& repF,
    const MatType& embedding,
    const std::vector<size_t>& oldFromNew,
    const size_t dof,
    const double theta)
    : sumQ(sumQ), repF(repF), embedding(embedding), oldFromNew(oldFromNew),
      dof(dof), theta(theta)
{
  /* Nothing To Do Here */
}

template <typename MatType>
double TSNERules<MatType>::BaseCase(
    const size_t queryIndex, const size_t referenceIndex)
{
  const VecType& queryPoint = embedding.col(oldFromNew[queryIndex]);
  const VecType& referencePoint = embedding.col(oldFromNew[referenceIndex]);
  const double distanceSq = (double)DistanceType::Evaluate(queryPoint,
                                                           referencePoint);

  if (distanceSq > arma::datum::eps)
  {
    double q = (double)dof / (dof + distanceSq);
    if (dof != 1)
      q = std::pow(q, (1.0 + dof) / 2.0);

    sumQ += q;
    repF.col(oldFromNew[queryIndex]) += q * q * (queryPoint - referencePoint);
  }

  return distanceSq;
}

template <typename MatType>
template <typename TreeType>
double TSNERules<MatType>::Score(
    const size_t queryIndex, TreeType& referenceNode)
{
  const VecType& queryPoint = embedding.col(oldFromNew[queryIndex]);
  const VecType& referencePoint = referenceNode.Stat().Centroid();
  const double distanceSq = std::max(arma::datum::eps,
      (double)DistanceType::Evaluate(queryPoint, referencePoint));

  const double diameterSq = (double)referenceNode.Bound().Diameter();
  const double score = diameterSq / distanceSq;
  if (score < theta * theta)
  {
    double q = (double)dof / (dof + distanceSq);
    if (dof != 1)
      q = std::pow(q, (1.0 + dof) / 2.0);

    sumQ += q * referenceNode.NumDescendants();
    repF.col(oldFromNew[queryIndex]) += q * q *
        referenceNode.NumDescendants() * (queryPoint - referencePoint);

    return DBL_MAX;
  }
  else
  {
    return score;
  }
}

template <typename MatType>
template <typename TreeType>
double TSNERules<MatType>::Rescore(
    const size_t queryIndex, TreeType& referenceNode, const double oldScore)
{
  return oldScore;
}

template <typename MatType>
template <typename TreeType>
double TSNERules<MatType>::Score(
    TreeType& queryNode, TreeType& referenceNode)
{
  const VecType& queryPoint = queryNode.Stat().Centroid();
  const VecType& referencePoint = referenceNode.Stat().Centroid();
  const double distanceSq = std::max(arma::datum::eps,
      (double)DistanceType::Evaluate(queryPoint, referencePoint));

  const double diameterSq = (double)std::max(
      queryNode.Bound().Diameter(), referenceNode.Bound().Diameter());
  const double score = diameterSq / distanceSq;
  if (score < theta * theta)
  {
    double q = (double)dof / (dof + distanceSq);
    if (dof != 1)
      q = std::pow(q, (1.0 + dof) / 2.0);

    sumQ += q * queryNode.NumDescendants() * referenceNode.NumDescendants();
    for (size_t i = 0; i < queryNode.NumDescendants(); i++)
    {
      repF.col(oldFromNew[queryNode.Descendant(i)]) += q * q *
          referenceNode.NumDescendants() * (queryPoint - referencePoint);
    }
    for (size_t i = 0; i < referenceNode.NumDescendants(); i++)
    {
      repF.col(oldFromNew[referenceNode.Descendant(i)]) +=  q * q *
          queryNode.NumDescendants() * (referencePoint - queryPoint);
    }

    return DBL_MAX;
  }
  else
  {
    return score;
  }
}

template <typename MatType>
template <typename TreeType>
double TSNERules<MatType>::Rescore(
    TreeType& queryNode, TreeType& referenceNode, const double oldScore)
{
  return oldScore;
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_RULES_TSNE_RULES_IMPL_HPP
