/**
 * @file kmeans_selection.hpp
 * @author Marcus Edel
 *
 * Use the centroids of the K-Means clustering method for use in the Nystroem
 * method of kernel matrix approximation.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
   * in the dataset.  You are responsible for deleting the returned matrix!
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
