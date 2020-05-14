/**
 * @file silhouette_score_impl.hpp
 * @author Khizir Siddiqui
 *
 * The implementation of the class SilhouetteScore.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_SILHOUETTE_SCORE_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_SILHOUETTE_SCORE_IMPL_HPP

#include <mlpack/core/cv/metrics/facilities.hpp>

namespace mlpack {
namespace cv {

template<typename DataType, typename Metric>
double SilhouetteScore::Overall(const DataType& X,
                                const arma::Row<size_t>& labels,
                                const Metric& metric)
{
  AssertSizes(data, labels, "SilhouetteScore::Overall()");
  DataType distances = PairwiseDistances(X, metric);
  return arma::mean(SamplesScore(distances, labels, metric));
}

template<typename DataType, typename Metric>
arma::rowvec SilhouetteScore::SamplesScore(const DataType& distances,
                                const arma::Row<size_t>& labels,
                                const Metric& metric)
{
  AssertSizes(data, labels, "SilhouetteScore::SamplesScore()");

  // Stores the silhouette scores of individual samples
  arma::rowvec sampleScores(distances.n_rows);
  // Finds one index per cluster.
  arma::ucolvec clusterLabels = arma::find_unique(labels);
  double interClusterDistance, intraClusterDistance;
  double minIntraClusterDistance = DBL_MAX;
  for (size_t i = 0; i < distances.n_rows; i++)
  {
    for (size_t j = 0; j < clusterLabels; j++)
    {
      if (labels(i) != clusterLabels(i) {
        intraClusterDistance = DistanceFromCluster(
          distances.col(i), labels, clusterLabels(i), metric, false
        );
        if (intraClusterDistance < minIntraClusterDistance) {
          minIntraClusterDistance = intraClusterDistance;
        }
      }
      else {
        interClusterDistance = DistanceFromCluster(
          distances.col(i), labels, clusterLabels(i), metric, true
        );
      }
    }
    sampleScores(i) = minIntraClusterDistance - interClusterDistance;
    sampleScores(i) /= std::max(intraClusterDistance, minIntraClusterDistance);
  }
  
  return sampleScores;
}

template<typename DataType, typename Metric>
double SilhouetteScore::DistanceFromCluster(const arma::rowvec& distances,
                                                  const arma::Row<size_t>& labels,
                                                  const size_t& elemLabel,
                                                  const Metric& metric,
                                                  const bool& sameCluster)
{
  AssertSizes(distances, labels, "SilhouetteScore::DistanceFromCluster()");

  // Find indices of elements with same label as subject element.
  arma::uvec sameClusterIndices = arma::find(labels == elemLabel);
  size_t numSameCluster = sameClusterIndices.n_elem;
  if (sameCluster == true && numSameCluster == 1)
  {
    // Return 0 if subject element is the only element in cluster.
    return 0.0;
  }
  else
  {
    double distance = arma::accu(distances.elem(sameClusterIndices));
    distance /= (numSameCluster - sameCluster);
    return distance;
  }
}

} // namespace cv
} // namespace mlpack

#endif
