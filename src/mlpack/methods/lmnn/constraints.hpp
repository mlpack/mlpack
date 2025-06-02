/**
 * @file methods/lmnn/constraints.hpp
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
template<typename MatType = arma::mat,
         typename LabelsType = arma::Row<size_t>,
         typename DistanceType = SquaredEuclideanDistance>
class Constraints
{
 public:
  //! Convenience typedef.
  using KNN = NeighborSearch<NearestNeighborSort, DistanceType, MatType>;

  // Convenience typedef for element type of data.
  using ElemType = typename MatType::elem_type;
  // Convenience typedef for column vector of data.
  using VecType = typename GetColType<MatType>::type;
  // Convenience typedef for cube of data.
  using CubeType = typename GetCubeType<MatType>::type;
  // Convenience typedef for dense matrix of indices.
  using UMatType = typename GetUDenseMatType<MatType>::type;
  // Convenience typedef for dense vector of indices.
  using UVecType = typename GetColType<UMatType>::type;

  /**
   * Constructor for creating a Constraints instance.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param k Number of target neighbors, impostors & triplets.
   */
  Constraints(const MatType& dataset,
              const LabelsType& labels,
              const size_t k);

  /**
   * Calculates k similar labeled nearest neighbors and stores them into the
   * passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store target neighbors.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param norms Input dataset norms.
   */
  void TargetNeighbors(UMatType& outputMatrix,
                       const MatType& dataset,
                       const LabelsType& labels,
                       const VecType& norms);

  /**
   * Calculates k similar labeled nearest neighbors for a batch of dataset and
   * stores them into the passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store target neighbors.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param norms Input dataset norms.
   * @param begin Index of the initial point of dataset.
   * @param batchSize Number of data points to use.
   */
  void TargetNeighbors(UMatType& outputMatrix,
                       const MatType& dataset,
                       const LabelsType& labels,
                       const VecType& norms,
                       const size_t begin,
                       const size_t batchSize);

  /**
   * Calculates k differently labeled nearest neighbors for each datapoint and
   * writes them back to passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store impostors.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param norms Input dataset norms.
   */
  void Impostors(UMatType& outputMatrix,
                 const MatType& dataset,
                 const LabelsType& labels,
                 const VecType& norms);

  /**
   * Calculates k differently labeled nearest neighbors & distances to
   * impostors for each datapoint and writes them back to passed matrices.
   *
   * @param outputNeighbors Coordinates matrix to store impostors.
   * @param outputDistance matrix to store distance.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param norms Input dataset norms.
   */
  void Impostors(UMatType& outputNeighbors,
                 MatType& outputDistance,
                 const MatType& dataset,
                 const LabelsType& labels,
                 const VecType& norms);

  /**
   * Calculates k differently labeled nearest neighbors for a batch of dataset
   * and writes them back to passed matrix.
   *
   * @param outputMatrix Coordinates matrix to store impostors.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param norms Input dataset norms.
   * @param begin Index of the initial point of dataset.
   * @param batchSize Number of data points to use.
   */
  void Impostors(UMatType& outputMatrix,
                 const MatType& dataset,
                 const LabelsType& labels,
                 const VecType& norms,
                 const size_t begin,
                 const size_t batchSize);

  /**
   * Calculates k differently labeled nearest neighbors & distances to
   * impostors for a batch of dataset and writes them back to passed matrices.
   *
   * @param outputNeighbors Coordinates matrix to store impostors.
   * @param outputDistance matrix to store distance.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param norms Input dataset norms.
   * @param begin Index of the initial point of dataset.
   * @param batchSize Number of data points to use.
   */
  void Impostors(UMatType& outputNeighbors,
                 MatType& outputDistance,
                 const MatType& dataset,
                 const LabelsType& labels,
                 const VecType& norms,
                 const size_t begin,
                 const size_t batchSize);

  /**
   * Calculates k differently labeled nearest neighbors & distances to
   * impostors for some points of dataset and writes them back to passed
   * matrices.
   *
   * @param outputNeighbors Coordinates matrix to store impostors.
   * @param outputDistance matrix to store distance.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param norms Input dataset norms.
   * @param points Indices of data points to calculate impostors on.
   * @param numPoints Number of points to actually calculate impostors on.
   */
  void Impostors(UMatType& outputNeighbors,
                 MatType& outputDistance,
                 const MatType& dataset,
                 const LabelsType& labels,
                 const VecType& norms,
                 const UVecType& points,
                 const size_t numPoints);

  /**
   * Generate triplets {i, j, l} for each datapoint i and writes back generated
   * triplets to matrix passed.
   *
   * @param outputMatrix Coordinates matrix to store triplets.
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param norms Input dataset norms.
   */
  void Triplets(UMatType& outputMatrix,
                const MatType& dataset,
                const LabelsType& labels,
                const VecType& norms);

  //! Get the number of target neighbors (k).
  const size_t& K() const { return k; }
  //! Modify the number of target neighbors (k).
  size_t& K() { return k; }

  //! Access the boolean value of precalculated.
  const bool& PreCalulated() const { return precalculated; }
  //! Modify the value of precalculated.
  bool& PreCalulated() { return precalculated; }

 private:
  //! Number of target neighbors & impostors to calulate.
  size_t k;

  //! Store unique labels.
  LabelsType uniqueLabels;

  //! Store indices of data points having similar label.
  std::vector<UVecType> indexSame;

  //! Store indices of data points having different label.
  std::vector<UVecType> indexDiff;

  //! False if nothing has ever been precalculated.
  bool precalculated;

  /**
  * Precalculate the unique labels, and indices of similar
  * and different datapoints on the basis of labels.
  */
  inline void Precalculate(const LabelsType& labels);

  /**
  * Re-order neighbors on the basis of increasing norm in case
  * of ties among distances.
  */
  inline void ReorderResults(const MatType& distances,
                             UMatType& neighbors,
                             const VecType& norms);
};

} // namespace mlpack

// Include implementation.
#include "constraints_impl.hpp"

#endif
