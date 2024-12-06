/**
 * @file methods/lmnn/lmnn_impl.hpp
 * @author Manish Kumar
 *
 * Implementation of Large Margin Nearest Neighbor class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_LMNN_IMPL_HPP
#define MLPACK_METHODS_LMNN_LMNN_IMPL_HPP

// In case it was not already included.
#include "lmnn.hpp"

namespace mlpack {

/**
 * Takes in a reference to the dataset. Copies the data, initializes
 * all of the member variables and constraint object and generate constraints.
 */
template<typename DistanceType, typename DeprecatedOptimizerType>
LMNN<DistanceType, DeprecatedOptimizerType>::LMNN(
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    const size_t k,
    const DistanceType distance) :
    dataset(&dataset),
    labels(&labels),
    k(k),
    regularization(0.5),
    updateInterval(1),
    distance(distance)
{ /* nothing to do */ }

template<typename DistanceType, typename DeprecatedOptimizerType>
LMNN<DistanceType, DeprecatedOptimizerType>::LMNN(
    const size_t k,
    const double regularization,
    const size_t updateInterval,
    const DistanceType distance) :
    k(k),
    regularization(regularization),
    updateInterval(updateInterval),
    distance(distance)
{ /* nothing to do */ }

template<typename DistanceType, typename DeprecatedOptimizerType>
template<typename... CallbackTypes, typename, typename>
void LMNN<DistanceType, DeprecatedOptimizerType>::LearnDistance(
    arma::mat& outputMatrix,
    CallbackTypes&&... callbacks)
{
  if (!dataset || !labels)
  {
    throw std::runtime_error("LMNN::LearnDistance(): cannot call without a "
        "dataset!");
  }

  LearnDistance(*dataset, *labels, outputMatrix, optimizer,
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename DistanceType, typename DeprecatedOptimizerType>
template<typename MatType,
         typename LabelsType,
         typename... CallbackTypes,
         typename /* SFINAE check that first callback is not an optimizer */,
         typename /* callback SFINAE check */>
void LMNN<DistanceType, DeprecatedOptimizerType>::LearnDistance(
    const MatType& dataset,
    const LabelsType& labels,
    MatType& outputMatrix,
    CallbackTypes&&... callbacks) const
{
  // This should be replaced with ens::StandardSGD when the deprecated members
  // are removed for mlpack 5.0.0.
  DeprecatedOptimizerType opt;
  LearnDistance(dataset, labels, outputMatrix, opt,
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename DistanceType, typename DeprecatedOptimizerType>
template<typename MatType,
         typename LabelsType,
         typename OptimizerType,
         typename... CallbackTypes,
         typename /* SFINAE check that opt is an ensmallen optimizer */>
void LMNN<DistanceType, DeprecatedOptimizerType>::LearnDistance(
    const MatType& dataset,
    const LabelsType& labels,
    MatType& outputMatrix,
    OptimizerType& opt,
    CallbackTypes&&... callbacks) const
{
  // LMNN objective function.
  LMNNFunction<MatType, LabelsType, DistanceType> objFunction(dataset, labels,
      k, regularization, updateInterval);

  // See if we were passed an initialized matrix. outputMatrix (L) must be
  // having r x d dimensionality.
  if ((outputMatrix.n_cols != dataset.n_rows) ||
      (outputMatrix.n_rows > dataset.n_rows) ||
      !(arma::is_finite(outputMatrix)))
  {
    outputMatrix.eye(dataset.n_rows, dataset.n_rows);
  }

  opt.Optimize(objFunction, outputMatrix, callbacks...);
}

// Serialize the LMNN object.
template<typename DistanceType, typename DeprecatedOptimizerType>
template<typename Archive>
void LMNN<DistanceType, DeprecatedOptimizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(k));
  ar(CEREAL_NVP(regularization));
  ar(CEREAL_NVP(updateInterval));
  ar(CEREAL_NVP(distance));
}

} // namespace mlpack

#endif
