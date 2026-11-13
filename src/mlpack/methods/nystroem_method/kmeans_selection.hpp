/**
 * @file methods/nystroem_method/kmeans_selection.hpp
 * @author Marcus Edel
 *
 * Use the centroids of the K-Means clustering method for use in the Nystroem
 * method of kernel matrix approximation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NYSTROEM_METHOD_KMEANS_SELECTION_HPP
#define MLPACK_METHODS_NYSTROEM_METHOD_KMEANS_SELECTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

namespace mlpack {

/**
 * Implementation of the kmeans sampling scheme.
 *
 * @tparam ClusteringType Type of clustering.
 * @tparam maxIterations Maximum number of iterations allowed before giving up.
 */
template<typename ClusteringType = KMeans<>, size_t maxIterations = 5>
class KMeansSelection
{
 public:
  /**
   * Use the K-Means clustering method to select the specified number of points
   * in the dataset.  You are responsible for deleting the returned matrix!
   *
   * KMeans currently only supports double-precision data, so if MatType is
   * not arma::mat, a double-precision copy of the data is clustered, and the
   * returned centroids are converted back to MatType.
   *
   * @param data Dataset to sample from.
   * @param m Number of points to select.
   * @return Matrix pointer in which centroids are stored.
   */
  template<typename MatType>
  static const MatType* Select(const MatType& data, const size_t m)
  {
    arma::Row<size_t> assignments;
    arma::mat centroids;

    // Perform the K-Means clustering method.
    ClusteringType kmeans(maxIterations);
    if constexpr (std::is_same_v<MatType, arma::mat>)
    {
      kmeans.Cluster(data, m, assignments, centroids);
      return new arma::mat(std::move(centroids));
    }
    else
    {
      kmeans.Cluster(arma::conv_to<arma::mat>::from(data), m, assignments,
          centroids);
      return new MatType(arma::conv_to<MatType>::from(centroids));
    }
  }
};

} // namespace mlpack

#endif
