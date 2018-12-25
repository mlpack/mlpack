/**
 * @file kde_rules_impl.hpp
 * @author Roberto Hueso (robertohueso96@gmail.com)
 *
 * Implementation of rules for Kernel Density Estimation with generic trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_KDE_RULES_IMPL_HPP
#define MLPACK_METHODS_KDE_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "kde_rules.hpp"

namespace mlpack {
namespace kde {

template<typename MetricType, typename KernelType, typename TreeType>
KDERules<MetricType, KernelType, TreeType>::KDERules(
    const arma::mat& referenceSet,
    const arma::mat& querySet,
    arma::vec& densities,
    const double relError,
    const double absError,
    const std::vector<size_t>& oldFromNewQueries,
    MetricType& metric,
    KernelType& kernel,
    const bool sameSet) :
    referenceSet(referenceSet),
    querySet(querySet),
    densities(densities),
    absError(absError),
    relError(relError),
    oldFromNewQueries(oldFromNewQueries),
    metric(metric),
    kernel(kernel),
    sameSet(sameSet),
    lastQueryIndex(querySet.n_cols),
    lastReferenceIndex(referenceSet.n_cols),
    baseCases(0),
    scores(0)
{
  // Nothing to do.
}

//! The base case.
template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline
double KDERules<MetricType, KernelType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // If reference and query sets are the same we don't want to compute the
  // estimation of a point with itself.
  if (sameSet && (queryIndex == referenceIndex))
    return 0.0;

  // Avoid duplicated calculations.
  if ((lastQueryIndex == queryIndex) && (lastReferenceIndex == referenceIndex))
    return 0.0;

  // Calculations.
  double distance = metric.Evaluate(querySet.col(queryIndex),
                                    referenceSet.col(referenceIndex));
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
    densities(oldFromNewQueries.at(queryIndex)) += kernel.Evaluate(distance);
  else
    densities(queryIndex) += kernel.Evaluate(distance);

  ++baseCases;
  lastQueryIndex = queryIndex;
  lastReferenceIndex = referenceIndex;
  return distance;
}

//! Single-tree scoring function.
template<typename MetricType, typename KernelType, typename TreeType>
double KDERules<MetricType, KernelType, TreeType>::
Score(const size_t queryIndex, TreeType& referenceNode)
{
  double score;
  bool newCalculations = true;
  const arma::vec& queryPoint = querySet.unsafe_col(queryIndex);
  const double minDistance = referenceNode.MinDistance(queryPoint);
  const double maxKernel = kernel.Evaluate(minDistance);
  const double minKernel =
      kernel.Evaluate(referenceNode.MaxDistance(queryPoint));
  const double bound = maxKernel - minKernel;

  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid &&
      lastQueryIndex == queryIndex &&
      traversalInfo.LastReferenceNode() != NULL &&
      traversalInfo.LastReferenceNode()->Point(0) == referenceNode.Point(0))
  {
    // Don't duplicate calculations.
    newCalculations = false;
    lastQueryIndex = queryIndex;
    lastReferenceIndex = referenceNode.Point(0);
  }

  if (bound <= (absError + relError * minKernel) / referenceSet.n_cols &&
      newCalculations)
  {
    double kernelValue;

    // Calculate kernel value based on reference node centroid.
    if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
    {
      kernelValue = EvaluateKernel(queryIndex, referenceNode.Point(0));
    }
    else
    {
      kde::KDEStat& referenceStat = referenceNode.Stat();
      if (!referenceStat.ValidCentroid())
      {
        arma::vec referenceCenter;
        referenceNode.Center(referenceCenter);
        referenceStat.SetCentroid(std::move(referenceCenter));
      }
      kernelValue = EvaluateKernel(queryPoint, referenceStat.Centroid());
    }

    // Add kernel value to density estimations
    if (tree::TreeTraits<TreeType>::RearrangesDataset)
    {
      densities(oldFromNewQueries.at(queryIndex)) +=
        referenceNode.NumDescendants() * kernelValue;
    }
    else
    {
      densities(queryIndex) += referenceNode.NumDescendants() * kernelValue;
    }
    // Don't explore this tree branch
    score = DBL_MAX;
  }
  else
  {
    score = minDistance;
  }

  ++scores;
  traversalInfo.LastReferenceNode() = &referenceNode;
  traversalInfo.LastScore() = score;
  return score;
}

template<typename MetricType, typename KernelType, typename TreeType>
double KDERules<MetricType, KernelType, TreeType>::Rescore(
    const size_t /* queryIndex */,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  // If it's pruned it continues to be pruned.
  return oldScore;
}

//! Double-tree scoring function.
template<typename MetricType, typename KernelType, typename TreeType>
inline double KDERules<MetricType, KernelType, TreeType>::
Score(TreeType& queryNode, TreeType& referenceNode)
{
  double score;
  // Calculations are not duplicated.
  bool newCalculations = true;
  const double minDistance = queryNode.MinDistance(referenceNode);
  const double maxKernel = kernel.Evaluate(minDistance);
  const double minKernel =
      kernel.Evaluate(queryNode.MaxDistance(referenceNode));
  const double bound = maxKernel - minKernel;

  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
  {
    if ((traversalInfo.LastQueryNode() != NULL) &&
        (traversalInfo.LastReferenceNode() != NULL) &&
        (traversalInfo.LastQueryNode()->Point(0) == queryNode.Point(0)) &&
        (traversalInfo.LastReferenceNode()->Point(0) == referenceNode.Point(0)))
    {
      // Don't duplicate calculations.
      newCalculations = false;
      lastQueryIndex = queryNode.Point(0);
      lastReferenceIndex = referenceNode.Point(0);
    }
  }

  // If possible, avoid some calculations because of the error tolerance
  if (bound <= (absError + relError * minKernel) / referenceSet.n_cols &&
      newCalculations)
  {
    // Auxiliary variables.
    double kernelValue;
    kde::KDEStat& referenceStat = referenceNode.Stat();
    kde::KDEStat& queryStat = queryNode.Stat();

    // If calculating a center is not required.
    if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
    {
      kernelValue = EvaluateKernel(queryNode.Point(0), referenceNode.Point(0));
    }
    // Sadly, we have no choice but to calculate the center.
    else
    {
      // Calculate center for each node if it has not been calculated yet.
      if (!referenceStat.ValidCentroid())
      {
        arma::vec referenceCenter;
        referenceNode.Center(referenceCenter);
        referenceStat.SetCentroid(std::move(referenceCenter));
      }
      if (!queryStat.ValidCentroid())
      {
        arma::vec queryCenter;
        queryNode.Center(queryCenter);
        queryStat.SetCentroid(std::move(queryCenter));
      }
      // Compute kernel value.
      kernelValue = EvaluateKernel(queryStat.Centroid(),
                                   referenceStat.Centroid());
    }

    // Can be paralellized but we avoid it for now because of a compilation
    // error in visual C++ compiler.
    // #pragma omp for
    for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
    {
      if (tree::TreeTraits<TreeType>::RearrangesDataset)
        densities(oldFromNewQueries.at(queryNode.Descendant(i))) +=
          referenceNode.NumDescendants() * kernelValue;
      else
        densities(queryNode.Descendant(i)) +=
          referenceNode.NumDescendants() * kernelValue;
    }
    score = DBL_MAX;
  }
  else
  {
    score = minDistance;
  }

  ++scores;
  traversalInfo.LastQueryNode() = &queryNode;
  traversalInfo.LastReferenceNode() = &referenceNode;
  traversalInfo.LastScore() = score;
  return score;
}

//! Double-tree
template<typename MetricType, typename KernelType, typename TreeType>
double KDERules<MetricType, KernelType, TreeType>::
Rescore(TreeType& /*queryNode*/,
        TreeType& /*referenceNode*/,
        const double oldScore) const
{
  return oldScore;
}

template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline double KDERules<MetricType, KernelType, TreeType>::
EvaluateKernel(const size_t queryIndex,
               const size_t referenceIndex) const
{
  return EvaluateKernel(querySet.unsafe_col(queryIndex),
                        referenceSet.unsafe_col(referenceIndex));
}

template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline double KDERules<MetricType, KernelType, TreeType>::
EvaluateKernel(const arma::vec& query, const arma::vec& reference) const
{
  return kernel.Evaluate(metric.Evaluate(query, reference));
}

} // namespace kde
} // namespace mlpack

#endif
