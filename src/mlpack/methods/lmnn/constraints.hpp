/**
 * @file constraints.hpp
 * @author Manish Kumar
 *
 * Declaration of the Constraints class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_CONSTRAINTS_HPP
#define MLPACK_METHODS_LMNN_CONSTRAINTS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

namespace mlpack {
namespace lmnn {
/**
 * Interface for generating distance based constraints on a given
 * dataset, provided corresponding true labels and a quantity parameter (k)
 * are specified.
 *
 * Class provides TargetNeighbors() (Used for calculating target neighbors
 * of each data point), Impostors() (used for calculating impostors of each
 * data point) and Triplets() (Generates sets of {dataset, target neighbors,
 * impostors} tripltets.)
 */
template<typename MetricType = metric::SquaredEuclideanDistance>
class Constraints
{
 public:
  //! Convenience typedef.
  typedef neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType>
      KNN;

  /**
   * Constructor for creating a Constraints instance.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param k Number of target neighbors, impostors & triplets.
   */
  Constraints(const arma::mat& dataset,
              const arma::Row<size_t>& labels,
              size_t k);

  /**
   * Calculates k similar labeled nearest neighbors and stores them into the
   * passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store target neighbors.
   */
  void TargetNeighbors(arma::Mat<size_t>& outputMatrix);

  /**
   * Calculates k similar labeled nearest neighbors for a batch of dataset and
   *  stores them into the passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store target neighbors.
   * @param begin Index of the initial point of dataset.
   * @param batchSize Number of data points to use.
   */
  void TargetNeighbors(arma::Mat<size_t>& outputMatrix,
                     const size_t begin,
                     const size_t batchSize);

  /**
   * Calculates k differently labeled nearest neighbors for each datapoint and
   * writes them back to passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store impostors.
   */
  void Impostors(arma::Mat<size_t>& outputMatrix);

  /**
   * Calculates k differently labeled nearest neighbors & distances to
   * impostors for each datapoint and writes them back to passed matrices.
   *
   * @param outputNeighbors Coordinates matrix to store impostors.
   * @param outputDistance matrix to store distance.
   */
  void Impostors(arma::Mat<size_t>& outputNeighbors,
                 arma::mat& outputDistance);

  /**
   * Calculates k differently labeled nearest neighbors for a batch of dataset
   * and writes them back to passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store impostors.
   * @param begin Index of the initial point of dataset.
   * @param batchSize Number of data points to use.
   */
  void Impostors(arma::Mat<size_t>& outputMatrix,
                 const size_t begin,
                 const size_t batchSize);

  /**
   * Calculates k differently labeled nearest neighbors & distances to
   * impostors for a batch of dataset and writes them back to passed matrices.
   *
   * @param outputNeighbors Coordinates matrix to store impostors.
   * @param outputDistance matrix to store distance.
   * @param begin Index of the initial point of dataset.
   * @param batchSize Number of data points to use.
   */
  void Impostors(arma::Mat<size_t>& outputNeighbors,
                 arma::mat& outputDistance,
                 const size_t begin,
                 const size_t batchSize);

  /**
   * Generate triplets {i, j, l} for each datapoint i and writes back generated
   * triplets to matrix passed.
   *
   * @param outputMatrix Coordinates matrix to store triplets.
   */
  void Triplets(arma::Mat<size_t>& outputMatrix);

  //! Get the dataset reference.
  const arma::mat& Dataset() const { return dataset; }
  //! Modify the dataset reference.
  arma::mat& Dataset() { return dataset; }

  //! Access the value of k.
  const size_t& K() const { return k; }
  //! Modify the value of k.
  size_t& K() { return k; }

 private:
  //! An alias of dataset.
  arma::mat dataset;

  //! An alias of Labels.
  arma::Row<size_t> labels;

  //! Number of target neighbors & impostors to calulate.
  size_t k;

  //! Store unique labels.
  arma::Row<size_t> uniqueLabels;

  //! Store indices of data points having similar label.
  std::vector<arma::uvec> indexSame;

  //! Store indices of data points having different label.
  std::vector<arma::uvec> indexDiff;

  /**
  * Precalculate the unique labels, and indices of similar
  * and different datapoints on the basis of labels.
  */
  void Precalculate();
};

} // namespace lmnn
} // namespace mlpack

// Include implementation.
#include "constraints_impl.hpp"

#endif
