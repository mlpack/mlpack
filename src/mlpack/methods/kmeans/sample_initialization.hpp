/**
 * @file sample_initialization.hpp
 * @author Ryan Curtin
 *
 * In order to construct initial centroids, randomly sample points from the
 * dataset.  This tends to give better results than the RandomPartition
 * strategy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_KMEANS_SAMPLE_INITIALIZATION_HPP
#define __MLPACK_METHODS_KMEANS_SAMPLE_INITIALIZATION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kmeans {

class SampleInitialization
{
 public:
  //! Empty constructor, required by the InitialPartitionPolicy type definition.
  SampleInitialization() { }

  /**
   * Initialize the centroids matrix by randomly sampling points from the data
   * matrix.
   *
   * @param data Dataset.
   * @param clusters Number of clusters.
   * @param centroids Matrix to put initial centroids into.
   */
  template<typename MatType>
  inline static void Cluster(const MatType& data,
                             const size_t clusters,
                             arma::mat& centroids)
  {
    centroids.set_size(data.n_rows, clusters);
    for (size_t i = 0; i < clusters; ++i)
    {
      // Randomly sample a point.
      const size_t index = math::RandInt(0, data.n_cols);
      centroids.col(i) = data.col(index);
    }
  }
};

} // namespace kmeans
} // namespace mlpack

#endif
