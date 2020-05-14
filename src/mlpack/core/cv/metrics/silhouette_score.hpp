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
namespace cv {

/**
 * 
 * //////////==============TODO=======================\\\\\\\\\\
 * 
 */
class SilhouetteScore
{
 public:
  /**
   * Run classification and calculate accuracy.
   *
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename DataType, typename Metric>
  static double Overall(const DataType& X,
                                     const arma::Row<size_t>& labels,
                                     const Metric& metric);

  template<typename DataType, typename Metric>
  static arma::rowvec SamplesScore(const DataType& distances,
                                     const arma::Row<size_t>& labels,
                                     const Metric& metric);

  template<typename DataType, typename Metric>
  static double DistanceFromCluster(const arma::rowvec& distances,
                                          const arma::Row<size_t>& labels,
                                          const size_t& label,
                                          const Metric& metric,
                                          const bool& sameCluster = 0);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to maximize the metric.
   */
  static const bool NeedsMinimization = false;
};

} // namespace cv
} // namespace mlpack

// Include implementation.
#include "silhouette_score_impl.hpp"

#endif
