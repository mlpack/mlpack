/**
 * @file methods/nca/nca.hpp
 * @author Ryan Curtin
 *
 * Declaration of NCA class (Neighborhood Components Analysis).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NCA_NCA_HPP
#define MLPACK_METHODS_NCA_NCA_HPP

#include <mlpack/core.hpp>

#include "nca_softmax_error_function.hpp"

namespace mlpack {

/**
 * An implementation of Neighborhood Components Analysis, both a linear
 * dimensionality reduction technique and a distance learning technique.  The
 * method seeks to improve k-nearest-neighbor classification on a dataset by
 * scaling the dimensions.  The method is nonparametric, and does not require a
 * value of k.  It works by using stochastic ("soft") neighbor assignments and
 * using optimization techniques over the gradient of the accuracy of the
 * neighbor assignments.
 *
 * For more details, see the following published paper:
 *
 * @code
 * @inproceedings{Goldberger2004,
 *   author = {Goldberger, Jacob and Roweis, Sam and Hinton, Geoff and
 *       Salakhutdinov, Ruslan},
 *   booktitle = {Advances in Neural Information Processing Systems 17},
 *   pages = {513--520},
 *   publisher = {MIT Press},
 *   title = {{Neighbourhood Components Analysis}},
 *   year = {2004}
 * }
 * @endcode
 */
template<typename DistanceType = SquaredEuclideanDistance,
         typename DeprecatedOptimizerType = ens::StandardSGD>
class NCA
{
 public:
  /**
   * Construct the Neighborhood Components Analysis object.  This simply stores
   * the reference to the dataset and labels as well as the parameters for
   * optimization before the actual optimization is performed.
   *
   * @param dataset Input dataset.
   * @param labels Input dataset labels.
   * @param distance Instantiated distance metric to use.
   */
  [[deprecated("Will be removed in mlpack 5.0.0.  Pass the dataset directly to "
               "LearnDistance() instead.")]]
  NCA(const arma::mat& dataset,
      const arma::Row<size_t>& labels,
      DistanceType distance = DistanceType());

  /**
   * Construct the Neighborhood Components Analysis object, optionally with an
   * instantiated distance metric.
   */
  NCA(DistanceType distance = DistanceType());

  /**
   * Perform Neighborhood Components Analysis.  The output distance learning
   * matrix is written into the passed reference.  If LearnDistance() is called
   * with an outputMatrix which has the correct size (dataset.n_rows x
   * dataset.n_rows), that matrix will be used as the starting point for
   * optimization.
   *
   * @tparam CallbackTypes Types of Callback functions.
   * @param outputMatrix Covariance matrix of Mahalanobis distance.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   */
  template<typename... CallbackTypes,
           typename = std::enable_if_t<IsEnsCallbackTypes<
               CallbackTypes...>::value>,
           typename = std::enable_if_t<
               !FirstElementIsArma<CallbackTypes...>::value>>
  [[deprecated("Will be removed in mlpack 5.0.0.  Use the version that takes a "
               "dataset as a parameter.")]]
  void LearnDistance(arma::mat& outputMatrix, CallbackTypes&&... callbacks);

  /**
   * Perform Neighborhood Components Analysis.  The output distance learning
   * matrix is written into the passed reference.  If LearnDistance() is called
   * with an outputMatrix which has the correct size (dataset.n_rows x
   * dataset.n_rows), that matrix will be used as the starting point for
   * optimization.
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
               SoftmaxErrorFunction<MatType, LabelsType, DistanceType>,
               MatType
           >::value>,
           typename = std::enable_if_t<
               IsEnsCallbackTypes<CallbackTypes...>::value>>
  void LearnDistance(const MatType& dataset,
                     const LabelsType& labels,
                     MatType& outputMatrix,
                     CallbackTypes&&... callbacks) const;

  /**
   * Perform Neighborhood Components Analysis.  The output distance learning
   * matrix is written into the passed reference.  If LearnDistance() is called
   * with an outputMatrix which has the correct size (dataset.n_rows x
   * dataset.n_rows), that matrix will be used as the starting point for
   * optimization.
   *
   * @param dataset Dataset to learn distance metric on.
   * @param labels Labels for dataset.
   * @param optimizer Instantiated ensmallen optimizer to use for NCA.
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
               SoftmaxErrorFunction<MatType, LabelsType, DistanceType>,
               MatType
           >::value>>
  void LearnDistance(const MatType& dataset,
                     const LabelsType& labels,
                     MatType& outputMatrix,
                     OptimizerType& optimizer,
                     CallbackTypes&&... callbacks) const;

  //! Get the dataset reference.
  [[deprecated("Will be removed in mlpack 5.0.0.")]]
  const arma::mat& Dataset() const { return *dataset; }
  //! Get the labels reference.
  [[deprecated("Will be removed in mlpack 5.0.0.")]]
  const arma::Row<size_t>& Labels() const { return *labels; }

  //! Get the optimizer.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use the LearnDistance() "
               "version that takes the optimizer as a parameter instead.")]]
  const DeprecatedOptimizerType& Optimizer() const { return optimizer; }
  //! Modify the optimizer.
  [[deprecated("Will be removed in mlpack 5.0.0.  Use the LearnDistance() "
               "version that takes the optimizer as a parameter instead.")]]
  DeprecatedOptimizerType& Optimizer() { return optimizer; }

  //! Get the distance.
  const DistanceType Distance() const { return distance; }
  //! Modify the distance.
  DistanceType& Distance() { return distance; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Dataset pointer (will be removed in mlpack 5.0.0).
  const arma::mat* dataset;
  //! Labels reference (will be removed in mlpack 5.0.0).
  const arma::Row<size_t>* labels;
  //! The optimizer to use (will be removed in mlpack 5.0.0).
  DeprecatedOptimizerType optimizer;

  //! Distance to be used.
  DistanceType distance;
};

} // namespace mlpack

// Include the implementation.
#include "nca_impl.hpp"

#endif
