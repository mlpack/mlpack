/**
 * @file methods/kde/kde_rules.hpp
 * @author Roberto Hueso
 *
 * Rules for Kernel Density Estimation, so that it can be done with arbitrary
 * tree types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KDE_RULES_HPP
#define MLPACK_METHODS_KDE_RULES_HPP

#include <mlpack/core/tree/traversal_info.hpp>

namespace mlpack {

/**
 * A dual-tree traversal Rules class for kernel density estimation.  This
 * contains the Score() and BaseCase() implementations.
 */
template<typename DistanceType, typename KernelType, typename TreeType>
class KDERules
{
 public:
  /**
   * Construct KDERules.
   *
   * @param referenceSet Reference set data.
   * @param querySet Query set data.
   * @param densities Vector where estimations will be written.
   * @param relError Relative error tolerance.
   * @param absError Absolute error tolerance.
   * @param mcProb Probability of relative error compliance for Monte Carlo
   *               estimations.
   * @param initialSampleSize Initial size of the Monte Carlo samples.
   * @param mcAccessCoef Access coefficient for Monte Carlo estimations.
   * @param mcBreakCoef Break coefficient for Monte Carlo estimations.
   * @param distance Instantiated distance metric.
   * @param kernel Instantiated kernel.
   * @param monteCarlo If true Monte Carlo estimations will be applied when
   *                   possible.
   * @param sameSet True if query and reference sets are the same
   *                (monochromatic evaluation).
   */
  KDERules(const arma::mat& referenceSet,
           const arma::mat& querySet,
           arma::vec& densities,
           const double relError,
           const double absError,
           const double mcProb,
           const size_t initialSampleSize,
           const double mcAccessCoef,
           const double mcBreakCoef,
           DistanceType& distance,
           KernelType& kernel,
           const bool monteCarlo,
           const bool sameSet);

  //! Base Case.
  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  //! SingleTree Rescore.
  double Score(const size_t queryIndex, TreeType& referenceNode);

  //! SingleTree Score.
  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore) const;

  //! Dual-Tree Score.
  double Score(TreeType& queryNode, TreeType& referenceNode);

  //! Dual-Tree Rescore.
  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore) const;

  using TraversalInfoType = mlpack::TraversalInfo<TreeType>;

  //! Get traversal information.
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }

  //! Modify traversal information.
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

  //! Get the number of base cases.
  size_t BaseCases() const { return baseCases; }

  //! Get the number of scores.
  size_t Scores() const { return scores; }

  //! Get the minimum number of base cases we need to perform to have acceptable
  //! results.
  size_t MinimumBaseCases() const { return 0; }

 private:
  //! Evaluate kernel value of 2 points given their indexes.
  double EvaluateKernel(const size_t queryIndex,
                        const size_t referenceIndex) const;

  //! Evaluate kernel value of 2 points.
  double EvaluateKernel(const arma::vec& query,
                        const arma::vec& reference) const;

  //! Calculate depth alpha for some node.
  double CalculateAlpha(TreeType* node);

  //! The reference set.
  const arma::mat& referenceSet;

  //! The query set.
  const arma::mat& querySet;

  //! Density values.
  arma::vec& densities;

  //! Absolute error tolerance.
  const double absError;

  //! Relatve error tolerance.
  const double relError;

  //! Significance level for relative error compliance for Monte Carlo
  //! estimations.
  const double mcBeta;

  //! Initial sample size for Monte Carlo estimations.
  const size_t initialSampleSize;

  //! Coefficient to control how much larger does the amount of node descendants
  //! has to be compared to the initial sample size in order to be a candidate
  //! for Monte Carlo estimations.
  const double mcAccessCoef;

  //! Coefficient to control what fraction of the amount of node's descendants
  //! is the limit before Monte Carlo estimation recurses.
  const double mcBreakCoef;

  //! Instantiated distance metric.
  DistanceType& distance;

  //! Instantiated kernel.
  KernelType& kernel;

  //! Whether Monte Carlo estimations are going to be applied.
  const bool monteCarlo;

  //! Accumulated not used MC alpha values for each query point.
  arma::vec accumMCAlpha;

  //! Accumulated not used error tolerance for each query point.
  arma::vec accumError;

  //! Whether reference and query sets are the same.
  const bool sameSet;

  //! Whether the kernel used for the rule is the Gaussian Kernel.
  constexpr static bool kernelIsGaussian =
      std::is_same_v<KernelType, GaussianKernel>;

  //! Absolute error tolerance available for each reference point.
  const double absErrorTol;

  //! The last query index.
  size_t lastQueryIndex;

  //! The last reference index.
  size_t lastReferenceIndex;

  //! Traversal information.
  TraversalInfoType traversalInfo;

  //! The number of base cases.
  size_t baseCases;

  //! The number of scores.
  size_t scores;
};

/**
 * A dual-tree traversal Rules class for cleaning used trees before performing
 * kernel density estimation.
 */
template<typename TreeType>
class KDECleanRules
{
 public:
  //! Construct KDECleanRules.
  KDECleanRules() { /* Nothing to do. */ }

  //! Base Case.
  double BaseCase(const size_t /* queryIndex */, const size_t /* refIndex */);

  //! SingleTree Score.
  double Score(const size_t /* queryIndex */, TreeType& referenceNode);

  //! SingleTree Rescore.
  double Rescore(const size_t /* queryIndex */,
                 TreeType& /* referenceNode */,
                 const double oldScore) const { return oldScore; }

  //! Dual-Tree Score.
  double Score(TreeType& queryNode, TreeType& referenceNode);

  //! Dual-Tree Rescore.
  double Rescore(TreeType& /* queryNode */,
                 TreeType& /* referenceNode*/ ,
                 const double oldScore) const { return oldScore; }

  using TraversalInfoType = mlpack::TraversalInfo<TreeType>;

  //! Get traversal information.
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }

  //! Modify traversal information.
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  //! Traversal information.
  TraversalInfoType traversalInfo;
};

} // namespace mlpack

// Include implementation.
#include "kde_rules_impl.hpp"

#endif
