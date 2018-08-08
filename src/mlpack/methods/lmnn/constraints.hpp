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
#include "lmnn_targets_and_impostors_rules.hpp"
#include "lmnn_impostors_rules.hpp"
#include "lmnn_stat.hpp"

namespace mlpack {
namespace lmnn {

/**
 * Interface for generating distance based constraints on a given
 * dataset, provided corresponding true labels and a quantity parameter (k)
 * are specified.
 *
 * Class provides NeighborsAndImpostors() (used for calculating target neighbors
 * and impostors of each data point simultaneously) and Impostors() (used for
 * calculating impostors of each data point).
 */
template<typename MetricType = metric::SquaredEuclideanDistance>
class Constraints
{
 public:
  typedef tree::KDTree<MetricType, LMNNStat, arma::mat> TreeType;

  /**
   * Constructor for creating a Constraints instance.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param k Number of target neighbors, impostors & triplets.
   */
  Constraints(const arma::mat& dataset,
              const arma::Row<size_t>& labels,
              const size_t k);

  /**
   * Free all memory.
   */
  ~Constraints();

  /**
   * Calculates neighborsK similar labeled nearest neighbors and impostorsK
   * impostors and stores them into the passed matrices.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param neighborsK Number of neighbors to search for.
   * @param impostorsK Number of impostors to search for.
   * @param neighbors Matrix to output target neighbors into.
   * @param impostors Matrix to output impostors into.
   */
  void TargetsAndImpostors(const arma::mat& dataset,
                           const arma::Row<size_t>& labels,
                           const size_t neighborsK,
                           const size_t impostorsK,
                           arma::Mat<size_t>& neighbors,
                           arma::Mat<size_t>& impostors);

  /**
   * Calculates k differently labeled nearest neighbors for each datapoint and
   * writes them back to passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store impostors.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   */
  void Impostors(arma::Mat<size_t>& outputMatrix,
                 const arma::mat& dataset,
                 const arma::Row<size_t>& labels,
                 const arma::mat& transformation);

  /**
   * Calculates k differently labeled nearest neighbors & distances to
   * impostors for each datapoint and writes them back to passed matrices.
   *
   * @param outputNeighbors Coordinates matrix to store impostors.
   * @param outputDistance matrix to store distance.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   */
  void Impostors(arma::Mat<size_t>& outputNeighbors,
                 arma::mat& outputDistance,
                 const arma::mat& dataset,
                 const arma::Row<size_t>& labels,
                 const arma::mat& transformation);

  /**
   * Calculates k differently labeled nearest neighbors for a batch of dataset
   * and writes them back to passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store impostors.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param begin Index of the initial point of dataset.
   * @param batchSize Number of data points to use.
   */
  void Impostors(arma::Mat<size_t>& outputMatrix,
                 const arma::mat& dataset,
                 const arma::Row<size_t>& labels,
                 const size_t begin,
                 const size_t batchSize,
                 const arma::mat& transformation);

  /**
   * Calculates k differently labeled nearest neighbors & distances to
   * impostors for a batch of dataset and writes them back to passed matrices.
   *
   * @param outputNeighbors Coordinates matrix to store impostors.
   * @param outputDistance matrix to store distance.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param begin Index of the initial point of dataset.
   * @param batchSize Number of data points to use.
   */
  void Impostors(arma::Mat<size_t>& outputNeighbors,
                 arma::mat& outputDistance,
                 const arma::mat& dataset,
                 const arma::Row<size_t>& labels,
                 const size_t begin,
                 const size_t batchSize,
                 const arma::mat& transformation);

  //! Get the number of target neighbors (k).
  const size_t& K() const { return k; }

  //! Access the boolean value of precalculated.
  const bool& PreCalulated() const { return precalculated; }
  //! Modify the value of precalculated.
  bool& PreCalulated() { return precalculated; }

 private:
  //! Number of target neighbors & impostors to calulate.
  const size_t k;

  //! Store unique labels.
  arma::Row<size_t> uniqueLabels;

  //! Store indices of data points having similar label.
  std::vector<arma::uvec> indexSame;

  //! Store indices of data points having different label.
  std::vector<arma::uvec> indexDiff;

  //! Reference tree used for search.
  TreeType* tree;
  //! Sorted labels for points in the reference tree.
  arma::Row<size_t> sortedLabels;
  //! Mapping used in tree building.
  std::vector<size_t> oldFromNew;
  //! Mapping used in tree building.
  std::vector<size_t> newFromOld;

  //! False if nothing has ever been precalculated.
  bool precalculated;

  /**
   * Precalculate the unique labels, and indices of similar
   * and different datapoints on the basis of labels.
   */
  inline void Precalculate(const arma::Row<size_t>& labels);

  /**
   * Compute the impostors of the given set.
   */
  void ComputeImpostors(const arma::mat& referenceSet,
                        const arma::Row<size_t>& referenceLabels,
                        const arma::mat& querySet,
                        const arma::Row<size_t>& queryLabels,
                        const arma::mat& transformation,
                        arma::Mat<size_t>& neighbors,
                        arma::mat& distances) const;
};

} // namespace lmnn
} // namespace mlpack

// Include implementation.
#include "constraints_impl.hpp"

#endif
