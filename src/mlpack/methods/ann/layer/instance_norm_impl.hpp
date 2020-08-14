/**
 * @file methods/ann/layer/instance_norm_impl.hpp
 * @author Anjishnu Mukherjee
 *
 * Implementation of the Instance Normalization Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_INSTANCENORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_INSTANCENORM_IMPL_HPP

// In case it is not included.
#include "instance_norm.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType>
InstanceNorm<InputDataType, OutputDataType>::InstanceNorm() :
    size(0),
    eps(1e-8),
    average(true),
    momentum(0.0),
    deterministic(false),
    reset(false)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
InstanceNorm<InputDataType, OutputDataType>::InstanceNorm(
    const size_t size,
    const double eps,
    const bool average,
    const double momentum) :
    size(size),
    eps(eps),
    average(average),
    momentum(momentum),
    deterministic(false),
    reset(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void InstanceNorm<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input,
    arma::Mat<eT>& output)
{
  // Instance Norm with (N, C, H, W) is same as Batch Norm with (1, N*C, H, W),
  // where N is the batchSize, C is the number of channels, H and W are the
  // height and width of each image respectively.
  if(!reset)
  {
    shapeA = input.n_rows;
    shapeB = input.n_cols;
    batchNorm = ann::BatchNorm<> (size*input.n_cols, eps, average, momentum);
    batchNorm.Reset();
    runningMean.zeros(size, 1);
    runningVariance.ones(size, 1);
    runningVariance = batchNorm.TrainingVariance();
    reset = true;
  }

  if (deterministic)
    batchNorm.Deterministic() = true;

  arma::mat inputCopy(const_cast<arma::Mat<eT>&>(input).memptr(), shapeA*shapeB,
        1, false, false);
  batchNorm.Forward(inputCopy, output);
  output.reshape(shapeA, shapeB);
  arma::mat runningTemp = arma::zeros(size, 1);
  runningMean = batchNorm.TrainingMean();
  runningMean.reshape(size, shapeB);
  runningTemp = arma::mean(runningMean, 1);
  runningMean.set_size(size, 1);
  runningMean = runningTemp;
  runningVariance = batchNorm.TrainingVariance();
  runningVariance.reshape(size, shapeB);
  runningTemp = arma::mean(runningVariance, 1);
  runningVariance.set_size(size, 1);
  runningVariance = runningTemp;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void InstanceNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  arma::mat inputCopy(const_cast<arma::Mat<eT>&>(input).memptr(), shapeA*shapeB,
      1, false, false);
  arma::mat gyCopy(const_cast<arma::Mat<eT>&>(gy).memptr(), shapeA*shapeB,
      1, false, false);
  batchNorm.Backward(inputCopy, gyCopy, g);
  g.reshape(shapeA, shapeB);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void InstanceNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  arma::mat errorCopy(const_cast<arma::Mat<eT>&>(error).memptr(), shapeA*shapeB,
      1, false, false);
  batchNorm.Gradient(input, errorCopy, gradient);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void InstanceNorm<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(size);
  ar & BOOST_SERIALIZATION_NVP(eps);
  ar & BOOST_SERIALIZATION_NVP(average);
  ar & BOOST_SERIALIZATION_NVP(momentum);
  ar & BOOST_SERIALIZATION_NVP(deterministic);
  ar & BOOST_SERIALIZATION_NVP(runningMean);
  ar & BOOST_SERIALIZATION_NVP(runningVariance);
  ar & BOOST_SERIALIZATION_NVP(reset);
  ar & BOOST_SERIALIZATION_NVP(shapeA);
  ar & BOOST_SERIALIZATION_NVP(shapeB);

  if (version > 0)
    ar & BOOST_SERIALIZATION_NVP(batchNorm);
}

} // namespace ann
} // namespace mlpack

#endif
