/**
 * @file kde_rules_impl.hpp
 * @author Roberto Hueso
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

// Used for Monte Carlo estimation.
#include <boost/math/distributions/normal.hpp>

namespace mlpack {
namespace kde {

template<typename MetricType, typename KernelType, typename TreeType>
KDERules<MetricType, KernelType, TreeType>::KDERules(
    const arma::mat& referenceSet,
    const arma::mat& querySet,
    arma::vec& densities,
    const double relError,
    const double absError,
    const double MCProb,
    const size_t initialSampleSize,
    const double MCAccessCoef,
    const double MCBreakCoef,
    MetricType& metric,
    KernelType& kernel,
    const bool monteCarlo,
    const bool sameSet) :
    referenceSet(referenceSet),
    querySet(querySet),
    densities(densities),
    absError(absError),
    relError(relError),
    MCBeta(1 - MCProb),
    initialSampleSize(initialSampleSize),
    MCAccessCoef(MCAccessCoef),
    MCBreakCoef(MCBreakCoef),
    metric(metric),
    kernel(kernel),
    monteCarlo(monteCarlo),
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
  const double distance = metric.Evaluate(querySet.col(queryIndex),
                                          referenceSet.col(referenceIndex));
  densities(queryIndex) += kernel.Evaluate(distance);

  ++baseCases;
  lastQueryIndex = queryIndex;
  lastReferenceIndex = referenceIndex;
  return distance;
}

//! Single-tree scoring function.
template<typename MetricType, typename KernelType, typename TreeType>
inline double KDERules<MetricType, KernelType, TreeType>::
Score(const size_t queryIndex, TreeType& referenceNode)
{
  kde::KDEStat& referenceStat = referenceNode.Stat();
  double score, maxKernel, minKernel, bound;
  const arma::vec& queryPoint = querySet.unsafe_col(queryIndex);
  const double minDistance = referenceNode.MinDistance(queryPoint);
  bool newCalculations = true;

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
  else
  {
    // Calculations are new.
    maxKernel = kernel.Evaluate(minDistance);
    minKernel = kernel.Evaluate(referenceNode.MaxDistance(queryPoint));
    bound = maxKernel - minKernel;
  }

  if (newCalculations &&
      bound <= (absError + relError * minKernel) / referenceSet.n_cols)
  {
    // Estimate values.
    double kernelValue;

    // Calculate kernel value based on reference node centroid.
    if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
      kernelValue = EvaluateKernel(queryIndex, referenceNode.Point(0));
    else
      kernelValue = EvaluateKernel(queryPoint, referenceStat.Centroid());

    densities(queryIndex) += referenceNode.NumDescendants() * kernelValue;

    // Don't explore this tree branch.
    score = DBL_MAX;
  }
  else if (monteCarlo &&
           referenceNode.NumDescendants() >= MCAccessCoef * initialSampleSize &&
           std::is_same<KernelType, kernel::GaussianKernel>::value)
  {
    // Calculate reference node depth.
    if (referenceStat.Depth() == 0)
      referenceStat.Depth() = CalculateDepth(referenceNode);

    // Monte Carlo probabilistic estimation.
    const double currentAlpha =
        MCBeta / std::pow(2, referenceNode.Stat().Depth() + 1);
    const boost::math::normal normalDist;
    const double z =
        std::abs(boost::math::quantile(normalDist, currentAlpha / 2));
    const size_t numDesc = referenceNode.NumDescendants();
    arma::vec sample;
    size_t m = initialSampleSize;
    double meanSample;
    bool useMonteCarloPredictions = true;
    while (m > 0)
    {
      const size_t oldSize = sample.size();
      const size_t newSize = oldSize + m;

      // Don't use probabilistic estimation if this is gonna take a close
      // amount of computation to the exact calculation.
      if (newSize >= MCBreakCoef * numDesc)
      {
        useMonteCarloPredictions = false;
        break;
      }

      // Increase the sample size.
      sample.resize(newSize);
      for (size_t i = 0; i < m; ++i)
      {
        // Sample and evaluate random points from the reference node.
        const size_t randomPoint = math::RandInt(0, numDesc);
        sample(oldSize + i) =
            EvaluateKernel(queryIndex, referenceNode.Descendant(randomPoint));
      }
      meanSample = arma::mean(sample);
      const double stddev = arma::stddev(sample);
      const double mThreshBase =
          z * stddev * (1 + relError) / (relError * meanSample);
      const size_t mThresh = std::ceil(mThreshBase * mThreshBase);

      if (sample.size() < mThresh)
        m = mThresh - sample.size();
      else
        m = 0;
    }

    if (useMonteCarloPredictions)
    {
      // Use Monte Carlo estimation.
      score = DBL_MAX;
      densities(queryIndex) += numDesc * meanSample;
    }
    else
    {
      // Recurse.
      score = minDistance;
    }
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
inline double KDERules<MetricType, KernelType, TreeType>::Rescore(
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
  double score, maxKernel, minKernel, bound;
  const double minDistance = queryNode.MinDistance(referenceNode);
  // Calculations are not duplicated.
  bool newCalculations = true;

  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid &&
      (traversalInfo.LastQueryNode() != NULL) &&
      (traversalInfo.LastReferenceNode() != NULL) &&
      (traversalInfo.LastQueryNode()->Point(0) == queryNode.Point(0)) &&
      (traversalInfo.LastReferenceNode()->Point(0) == referenceNode.Point(0)))
  {
    // Don't duplicate calculations.
    newCalculations = false;
    lastQueryIndex = queryNode.Point(0);
    lastReferenceIndex = referenceNode.Point(0);
  }
  else
  {
    // Calculations are new.
    maxKernel = kernel.Evaluate(minDistance);
    minKernel = kernel.Evaluate(queryNode.MaxDistance(referenceNode));
    bound = maxKernel - minKernel;
  }

  // If possible, avoid some calculations because of the error tolerance.
  if (newCalculations &&
      bound <= (absError + relError * minKernel) / referenceSet.n_cols)
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
      kernelValue = EvaluateKernel(queryStat.Centroid(),
                                   referenceStat.Centroid());
    }

    // Sum up estimations.
    for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
    {
      densities(queryNode.Descendant(i)) +=
          referenceNode.NumDescendants() * kernelValue;
    }
    score = DBL_MAX;
  }
  else if (monteCarlo &&
           referenceNode.NumDescendants() >= MCAccessCoef * initialSampleSize &&
           std::is_same<KernelType, kernel::GaussianKernel>::value)
  {
    // Calculate reference node depth.
    kde::KDEStat& referenceStat = referenceNode.Stat();
    if (referenceStat.Depth() == 0)
      referenceStat.Depth() = CalculateDepth(referenceNode);

    // Monte Carlo probabilistic estimation.
    const double currentAlpha =
        MCBeta / std::pow(2, referenceNode.Stat().Depth() + 1);
    const boost::math::normal normalDist;
    const double z =
        std::abs(boost::math::quantile(normalDist, currentAlpha / 2));
    const size_t numDesc = referenceNode.NumDescendants();
    arma::vec sample;
    arma::vec means = arma::zeros(queryNode.NumDescendants());
    size_t m;
    double meanSample;
    bool useMonteCarloPredictions = true;
    for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
    {
      const size_t queryIndex = queryNode.Descendant(i);
      sample.clear();
      m = initialSampleSize;
      while (m > 0)
      {
        const size_t oldSize = sample.size();
        const size_t newSize = oldSize + m;

        // Don't use probabilistic estimation if this is gonna take a close
        // amount of computation to the exact calculation.
        if (newSize >= MCBreakCoef * numDesc)
        {
          useMonteCarloPredictions = false;
          break;
        }

        // Increase the sample size.
        sample.resize(newSize);
        for (size_t i = 0; i < m; ++i)
        {
          // Sample and evaluate random points from the reference node.
          const size_t randomPoint = math::RandInt(0, numDesc);
          sample(oldSize + i) =
              EvaluateKernel(queryIndex, referenceNode.Descendant(randomPoint));
        }
        meanSample = arma::mean(sample);
        const double stddev = arma::stddev(sample);
        const double mThreshBase =
            z * stddev * (1 + relError) / (relError * meanSample);
        const size_t mThresh = std::ceil(mThreshBase * mThreshBase);

        if (sample.size() < mThresh)
          m = mThresh - sample.size();
        else
          m = 0;
      }

      if (useMonteCarloPredictions)
        means(i) = meanSample;
      else
        break;
    }

    if (useMonteCarloPredictions)
    {
      // Use Monte Carlo estimation.
      score = DBL_MAX;
      for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
        densities(queryNode.Descendant(i)) += numDesc * means(i);
    }
    else
    {
      // Recurse.
      score = minDistance;
    }
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

//! Double-tree rescore.
template<typename MetricType, typename KernelType, typename TreeType>
inline double KDERules<MetricType, KernelType, TreeType>::
Rescore(TreeType& /*queryNode*/,
        TreeType& /*referenceNode*/,
        const double oldScore) const
{
  // If a branch is pruned then it continues to be pruned.
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

template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline size_t KDERules<MetricType, KernelType, TreeType>::
CalculateDepth(const TreeType& node) const
{
  size_t depth = 0;
  const TreeType* currentNode = &node;
  while (currentNode != NULL)
  {
    ++depth;
    currentNode = currentNode->Parent();
  }
  return depth;
}

} // namespace kde
} // namespace mlpack

#endif
