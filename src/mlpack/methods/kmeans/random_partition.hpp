/**
 * @file random_partition.hpp
 * @author Ryan Curtin
 *
 * Very simple partitioner which partitions the data randomly into the number of
 * desired clusters.  Used as the default InitialPartitionPolicy for KMeans.
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
#ifndef __MLPACK_METHODS_KMEANS_RANDOM_PARTITION_HPP
#define __MLPACK_METHODS_KMEANS_RANDOM_PARTITION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kmeans {

/**
 * A very simple partitioner which partitions the data randomly into the number
 * of desired clusters.  It has no parameters, and so an instance of the class
 * is not even necessary.
 */
class RandomPartition
{
 public:
  //! Empty constructor, required by the InitialPartitionPolicy policy.
  RandomPartition() { }

  /**
   * Partition the given dataset into the given number of clusters.  Assignments
   * are random, and the number of points in each cluster should be equal (or
   * approximately equal).
   *
   * @tparam MatType Type of data (arma::mat or arma::sp_mat).
   * @param data Dataset to partition.
   * @param clusters Number of clusters to split dataset into.
   * @param assignments Vector to store cluster assignments into.  Values will
   *     be between 0 and (clusters - 1).
   */
  template<typename MatType>
  inline static void Cluster(const MatType& data,
                             const size_t clusters,
                             arma::Col<size_t>& assignments)
  {
    // Implementation is so simple we'll put it here in the header file.
    assignments = arma::shuffle(arma::linspace<arma::Col<size_t> >(0,
        (clusters - 1), data.n_cols));
  }
};

};
};

#endif
