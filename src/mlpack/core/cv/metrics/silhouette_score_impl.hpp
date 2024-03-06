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

template<typename DataType, typename Metric>
double SilhouetteScore::Overall(const DataType& X,
                                const arma::Row<size_t>& labels,
                                const Metric& metric)
{
  util::CheckSameSizes(X, labels, "SilhouetteScore::Overall()");
  return arma::mean(SamplesScore(X, labels, metric));
}

template<typename DataType>
arma::rowvec SilhouetteScore::SamplesScore(const DataType& distances,
                                           const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(distances, labels, "SilhouetteScore::SamplesScore()");

  // Stores the silhouette scores of individual samples.
  arma::rowvec sampleScores(distances.n_rows);
  // Finds one index per cluster.
  arma::ucolvec clusterLabels = arma::find_unique(labels, false);

  for (size_t i = 0; i < distances.n_rows; i++)
  {
    double interClusterDistance = DBL_MAX, intraClusterDistance = 0;
    double minInterClusterDistance = DBL_MAX;
    for (size_t j = 0; j < clusterLabels.n_elem; j++)
    {
      size_t clusterLabel = labels(clusterLabels(j));
      if (labels(i) != clusterLabel) {
        interClusterDistance = MeanDistanceFromCluster(
          distances.col(i), labels, clusterLabel, false);
        if (interClusterDistance < minInterClusterDistance) {
          minInterClusterDistance = interClusterDistance;
        }
      } else {
        intraClusterDistance = MeanDistanceFromCluster(
          distances.col(i), labels, clusterLabel, true);
        if (intraClusterDistance == 0) {
          // s(i) = 0, no more calculation needed.
          break;
        }
      }
    }
    if (intraClusterDistance == 0) {
      // i is the only element in the cluster.
      sampleScores(i) = 0.0;
    } else {
      sampleScores(i) = minInterClusterDistance - intraClusterDistance;
      sampleScores(i) /= std::max(
        intraClusterDistance, minInterClusterDistance);
    }
  }
  return sampleScores;
}

template<typename DataType, typename Metric>
arma::rowvec SilhouetteScore::SamplesScore(const DataType& X,
                                           const arma::Row<size_t>& labels,
                                           const Metric& metric)
{
  util::CheckSameSizes(X, labels, "SilhouetteScore::SamplesScore()");
  DataType distances = PairwiseDistances(X, metric);
  return SamplesScore(distances, labels);
}

inline double SilhouetteScore::MeanDistanceFromCluster(
    const arma::colvec& distances,
    const arma::Row<size_t>& labels,
    const size_t& elemLabel,
    const bool& sameCluster)
{
  // Find indices of elements with same label as elemLabel.
  arma::uvec sameClusterIndices = arma::find(labels == elemLabel);

  // Numver of elements in the given cluster.
  size_t numSameCluster = sameClusterIndices.n_elem;
  if ((sameCluster == true) && (numSameCluster == 1))
  {
    // Return 0 if subject element is the only element in cluster.
    return 0.0;
  }
  else
  {
    double distance = accu(distances.elem(sameClusterIndices));
    distance /= (numSameCluster - sameCluster);
    return distance;
  }
}

} // namespace mlpack

#endif
