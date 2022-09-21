/**
 * @file silhouette_score.hpp
 * @author Khizir Siddiqui
 *
 * The Silhouette metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_SILHOUETTE_SCORE_HPP
#define MLPACK_CORE_CV_METRICS_SILHOUETTE_SCORE_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * The Silhouette Score is a metric of performance for clustering
 * that represents the quality of clusters made as a result.
 * It provides an indication of goodness of fit and therefore a measure of how
 * well unseen samples are likely to be predicted by the model, considering
 * the inter-cluster and intra-cluster dissimilarities.
 * Silhoutte Score is dependent on the metric used to calculate the
 * dissimilarities. The best possible score is @f$ s(i) = 1.0 @f$.
 * Smaller values of Silhouette Score indicate poor clustering.
 *  Negative values would occur when a wrong label was put on the element.
 *  Values near zero indicate overlapping clusters.
 * For an element i @f$ a(i) @f$ is within cluster average dissimilarity
 * and @f$ b(i) @f$ is minimum of average dissimilarity from other clusters.
 * the Silhouette Score @f$ s(i) @f$ of a Sample is calculated by
 * @f{eqnarray*}{
 * s(i)  &=& \frac{b(i) - a(i)}{max\{b(i), a(i)\}}
 * @f}
 *
 * The Overall Silhouette Score is the mean of individual silhoutte scores.
 */
class SilhouetteScore
{
 public:
  /**
   * Find the overall silhouette score.
   *
   * @param X Column-major data used for clustering.
   * @param labels Labels assigned to data by clustering.
   * @param metric Metric to be used to calculate dissimilarity.
   * @return (double) silhouette score.
   */
  template<typename DataType, typename Metric>
  static double Overall(const DataType& X,
                        const arma::Row<size_t>& labels,
                        const Metric& metric);

  /**
   * Find the individual silhouette scores for precomputted dissimilarites.
   *
   * @param distances Square matrix containing distances between data points.
   * @param labels Labels assigned to data by clustering.
   * @return (arma::rowvec) element-wise silhouette score.
   */
  template<typename DataType>
  static arma::rowvec SamplesScore(const DataType& distances,
                                   const arma::Row<size_t>& labels);

  /**
   * Find silhouette score of all individual elements.
   * (Distance not precomputed).
   *
   * @param X Column-major data used for clustering.
   * @param labels Labels assigned to data by clustering.
   * @param metric Metric to be used to calculate dissimilarity.
   * @return (arma::rowvec) element-wise silhouette score.
   */
  template<typename DataType, typename Metric>
  static arma::rowvec SamplesScore(const DataType& X,
                                   const arma::Row<size_t>& labels,
                                   const Metric& metric);

  /**
   * Find mean distance of element from a given cluster.
   *
   * @param distances colvec containing distances from other elements.
   * @param labels Labels assigned to data by clustering.
   * @param label label of the target cluster.
   * @param sameCluster true if calculating mean distance from same cluster.
   * @return (double) distance from the cluster.
   */
  static double MeanDistanceFromCluster(const arma::colvec& distances,
                                        const arma::Row<size_t>& labels,
                                        const size_t& label,
                                        const bool& sameCluster = false);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to maximize the metric.
   */
  static const bool NeedsMinimization = false;
};

} // namespace mlpack

// Include implementation.
#include "silhouette_score_impl.hpp"

#endif
