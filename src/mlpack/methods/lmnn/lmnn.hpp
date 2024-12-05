/**
 * @file methods/lmnn/lmnn.hpp
 * @author Manish Kumar
 *
 * Declaration of Large Margin Nearest Neighbor class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_LMNN_HPP
#define MLPACK_METHODS_LMNN_LMNN_HPP

#include <mlpack/core.hpp>

#include "constraints.hpp"
#include "lmnn_function.hpp"

namespace mlpack {

/**
 * An implementation of Large Margin nearest neighbor metric learning technique.
 * The method seeks to improve clustering & classification algorithms on
 * a dataset by transforming the dataset representation in a more convenient
 * form for them. It introduces the concept of target neighbors and impostors,
 * focusing on the idea that the distance between impostors and the perimeters
 * established by target neighbors should be large and that between target
 * neighbors and data point should be small. It requires the knowledge of
 * target neighbors beforehand. Moreover, target neighbors once initialized
 * remain same.
 *
 * For more details, see the following published paper:
 *
 * @code
 * @ARTICLE{weinberger09distance,
 *   author = {Weinberger, K.Q. and Saul, L.K.},
 *   title = {{Distance metric learning for large margin nearest neighbor
 *       classification}},
 *   journal = {The Journal of Machine Learning Research},
 *   year = {2009},
 *   volume = {10},
 *   pages = {207--244},
 *   publisher = {MIT Press}
 * }
 * @endcode
 *
 * @tparam DistanceType The type of distance metric to use for computation.
 * @tparam OptimizerType Optimizer to use for developing distance.
 */
template<typename DistanceType = SquaredEuclideanDistance,
         typename DeprecatedOptimizerType = ens::AMSGrad>
class LMNN
{
 public:
  /**
   * Initialize the LMNN object, passing a dataset (distance metric
   * is learned using this dataset) and labels. Initialization will copy
   * both dataset and labels matrices to internal copies.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param k Number of targets to consider.
   * @param distance Type of distance metric used for computation.
   */
  [[deprecated("Will be removed in mlpack 5.0.0.  Pass the dataset directly to "
               "LearnDistance() instead.")]]
  LMNN(const arma::mat& dataset,
       const arma::Row<size_t>& labels,
       const size_t k,
       const DistanceType distance = DistanceType());

  /**
   * Construct the LMNN object, optionally with an instantiated distance metric.
   *
   * @param k Number of target neighbors to consider.
   * @param regularization Penalty to apply to objective function.
   * @param updateInterval Number of iterations between each recomputation of
   *     true neighbors and impostors.
   * @param distance Instantiated distance metric for computation.
   */
  LMNN(const size_t k,
       const double regularization = 0.5,
       const size_t updateInterval = 1,
       DistanceType distance = DistanceType());


  /**
   * Perform Large Margin Nearest Neighbors metric learning. The output
   * distance matrix is written into the passed reference. If the
   * LearnDistance() is called with an outputMatrix with correct dimensions,
   * then that matrix will be used as the starting point for optimization.
   *
   * @tparam CallbackTypes Types of Callback functions.
   * @param outputMatrix Covariance matrix of Mahalanobis distance.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   */
  template<typename... CallbackTypes,
           typename = std::enable_if_t<IsEnsCallbackTypes<
               CallbackTypes...
           >::value>,
           typename = std::enable_if_t<
               !FirstElementIsArma<CallbackTypes...>::value>>
  [[deprecated("Will be removed in mlpack 5.0.0.  Use the version that takes a "
               "dataset as a parameter.")]]
  void LearnDistance(arma::mat& outputMatrix, CallbackTypes&&... callbacks);

  /**
   * Perform Large Margin Nearest Neighbors metric learning. The output
   * distance matrix is written into the passed reference. If the
   * LearnDistance() is called with an outputMatrix with correct dimensions,
   * then that matrix will be used as the starting point for optimization.
   *
   * @param dataset Dataset to learn distance metric on.
   * @param labels Labels for dataset.
   * @param outputMatrix Covariance matrix of Mahalanobis distance.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   */
  template<typename MatType,
           typename LabelsType,
           typename... CallbackTypes,
           typename = std::enable_if_t<!IsEnsOptimizer<
               typename First<CallbackTypes...>::type,
               LMNNFunction<MatType, LabelsType, DistanceType>,
               MatType
           >::value>,
           typename = std::enable_if_t<IsEnsCallbackTypes<
               CallbackTypes...
           >::value>>
  void LearnDistance(const MatType& dataset,
                     const LabelsType& labels,
                     MatType& outputMatrix,
                     CallbackTypes&&... callbacks) const;

  /**
   * Perform Large Margin Nearest Neighbors metric learning. The output
   * distance matrix is written into the passed reference. If the
   * LearnDistance() is called with an outputMatrix with correct dimensions,
   * then that matrix will be used as the starting point for optimization.
   *
   * @param dataset Dataset to learn distance metric on.
   * @param labels Labels for dataset.
   * @param optimizer Instantiated ensmallen optimizer to use for LMNN.
   * @param outputMatrix Covariance matrix of Mahalanobis distance.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   */
  template<typename MatType,
           typename LabelsType,
           typename OptimizerType,
           typename... CallbackTypes,
           typename = std::enable_if_t<IsEnsOptimizer<
               OptimizerType,
               LMNNFunction<MatType, LabelsType, DistanceType>,
               MatType
           >::value>>
  void LearnDistance(const MatType& dataset,
                     const LabelsType& labels,
                     MatType& outputMatrix,
                     OptimizerType& optimizer,
                     CallbackTypes&&... callbacks) const;

  //! Get the dataset reference.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use the LearnDistance() "
               "version that takes the optimizer as a parameter instead.")]]
  const arma::mat& Dataset() const { return *dataset; }

  //! Get the labels reference.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use the LearnDistance() "
               "version that takes the optimizer as a parameter instead.")]]
  const arma::Row<size_t>& Labels() const { return *labels; }

  //! Access the regularization value.
  const double& Regularization() const { return regularization; }
  //! Modify the regularization value.
  double& Regularization() { return regularization; }

  //! Access the iteration update interval value.
  const size_t& UpdateInterval() const { return updateInterval; }
  //! Modify the iteration update interval value.
  size_t& UpdateInterval() { return updateInterval; }

  [[deprecated("Will be removed in mlpack 5.0.0.  Use UpdateInterval() "
               "instead.")]]
  const size_t& Range() const { return updateInterval; }
  [[deprecated("Will be removed in mlpack 5.0.0.  Use UpdateInterval() "
               "instead.")]]
  size_t& Range() { return updateInterval; }

  //! Access the value of k.
  const size_t& K() const { return k; }
  //! Modify the value of k.
  size_t K() { return k; }

  //! Get the optimizer.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use the LearnDistance() "
               "version that takes the optimizer as a parameter instead.")]]
  const DeprecatedOptimizerType& Optimizer() const { return optimizer; }
  //! Modify the optimizer.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use the LearnDistance() "
               "version that takes the optimizer as a parameter instead.")]]
  DeprecatedOptimizerType& Optimizer() { return optimizer; }

  // Serialize the LMNN object.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Dataset pointer (will be removed in mlpack 5.0.0).
  const arma::mat* dataset;
  //! Labels pointer (will be removed in mlpack 5.0.0).
  const arma::Row<size_t>* labels;

  //! Number of target points.
  size_t k;

  //! Regularization value.
  double regularization;

  //! Number of iterations after which impostors need to be recalculated.
  size_t updateInterval;

  //! Distance to be used.
  DistanceType distance;

  //! The optimizer to use (will be removed in mlpack 5.0.0).
  DeprecatedOptimizerType optimizer;
}; // class LMNN

} // namespace mlpack

// Include the implementation.
#include "lmnn_impl.hpp"

#endif
