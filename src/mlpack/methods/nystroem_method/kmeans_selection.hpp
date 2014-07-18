/**
 * @file kmeans_selection.hpp
 * @author Marcus Edel
 *
 * Use the centroids of the K-Means clustering method for use in the Nystroem
 * method of kernel matrix approximation.
 */
#ifndef __MLPACK_METHODS_NYSTROEM_METHOD_KMEANS_SELECTION_HPP
#define __MLPACK_METHODS_NYSTROEM_METHOD_KMEANS_SELECTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

namespace mlpack {
namespace kernel {

template<typename ClusteringType = kmeans::KMeans<> >
class KMeansSelection
{
 public:
  /**
   * Use the K-Means clustering method to select the specified number of points
   * in the dataset.
   *
   * @param data Dataset to sample from.
   * @param m Number of points to select.
   * @return Matrix pointer in which centroids are stored.
   */
  const static arma::mat* Select(const arma::mat& data,
                                 const size_t m,
                                 const size_t maxIterations = 5)
  {
    arma::Col<size_t> assignments;
    arma::mat* centroids = new arma::mat;

    // Perform the K-Means clustering method.
    ClusteringType kmeans(maxIterations);
    kmeans.Cluster(data, m, assignments, *centroids);

    return centroids;
  }
};

}; // namespace kernel
}; // namespace mlpack

#endif
