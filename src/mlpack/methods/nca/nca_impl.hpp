/**
 * @file methods/nca/nca_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated NCA class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NCA_NCA_IMPL_HPP
#define MLPACK_METHODS_NCA_NCA_IMPL_HPP

// In case it was not already included.
#include "nca.hpp"

namespace mlpack {

// Just set the internal matrix reference.
template<typename DistanceType, typename DeprecatedOptimizerType>
NCA<DistanceType, DeprecatedOptimizerType>::NCA(
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    DistanceType distance) :
    dataset(&dataset),
    labels(&labels),
    distance(std::move(distance))
{ /* Nothing to do. */ }

template<typename DistanceType, typename DeprecatedOptimizerType>
NCA<DistanceType, DeprecatedOptimizerType>::NCA(DistanceType distance) :
    distance(std::move(distance))
{ /* Nothing to do. */ }

template<typename DistanceType, typename DeprecatedOptimizerType>
template<typename... CallbackTypes,
         typename /* callback SFINAE check */,
         typename /* SFINAE check to disambiguate overloads */>
void NCA<DistanceType, DeprecatedOptimizerType>::LearnDistance(
    arma::mat& outputMatrix,
    CallbackTypes&&... callbacks)
{
  if (!dataset || !labels)
  {
    throw std::runtime_error("NCA::LearnDistance(): cannot call without a "
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
void NCA<DistanceType, DeprecatedOptimizerType>::LearnDistance(
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
void NCA<DistanceType, DeprecatedOptimizerType>::LearnDistance(
    const MatType& dataset,
    const LabelsType& labels,
    MatType& outputMatrix,
    OptimizerType& opt,
    CallbackTypes&&... callbacks) const
{
  SoftmaxErrorFunction<MatType, LabelsType, DistanceType> errorFunction(
      dataset, labels, distance);

  // See if we were passed an initialized matrix.
  if ((outputMatrix.n_rows != dataset.n_rows) ||
      (outputMatrix.n_cols != dataset.n_rows))
    outputMatrix.eye(dataset.n_rows, dataset.n_rows);

  opt.Optimize(errorFunction, outputMatrix,
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename DistanceType, typename DeprecatedOptimizerType>
template<typename Archive>
void NCA<DistanceType, DeprecatedOptimizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(distance));
}

} // namespace mlpack

#endif
