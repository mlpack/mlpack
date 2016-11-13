/**
 * @file kmeans_selection.hpp
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

#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

namespace mlpack {
namespace kernel {

/**
 * Implementation of the kmeans sampling scheme.
 *
 * @tparam ClusteringType Type of clustering.
 * @tparam maxIterations Maximum number of iterations allowed before giving up.
 */
template<typename ClusteringType = kmeans::KMeans<>, size_t maxIterations = 5>
class KMeansSelection
{
 public:
  /**
   * Use the K-Means clustering method to select the specified number of points
   * in the dataset.  You are responsible for deleting the returned matrix!
   *
   * @param data Dataset to sample from.
   * @param m Number of points to select.
   * @return Matrix pointer in which centroids are stored.
   */
  const static arma::mat* Select(const arma::mat& data, const size_t m)
  {
    arma::Row<size_t> assignments;
    arma::mat* centroids = new arma::mat;

    // Perform the K-Means clustering method.
    ClusteringType kmeans(maxIterations);
    kmeans.Cluster(data, m, assignments, *centroids);

    return centroids;
  }
};

} // namespace kernel
} // namespace mlpack

#endif
