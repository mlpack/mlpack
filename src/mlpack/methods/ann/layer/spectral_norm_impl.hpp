/**
 * @file spectral_norm_impl.hpp
 * @author Saksham Bansal
 *
 * Implementation of the SpectralNorm Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_SPECTRAL_NORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SPECTRAL_NORM_IMPL_HPP

// In case it is not included.
#include "spectral_norm.hpp"

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"
#include "../visitor/bias_set_visitor.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
SpectralNorm<InputDataType, OutputDataType, CustomLayers...>::SpectralNorm() :
    powerIterations(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
SpectralNorm<InputDataType, OutputDataType, CustomLayers...>::SpectralNorm(
    const size_t inSize,
    const size_t outSize,
    const size_t powerIterations) :
    powerIterations(powerIterations),
    wrappedLayer(new Linear<>(inSize, outSize))
{
  layerWeightSize = inSize * outSize + outSize;
  weights.set_size(layerWeightSize, 1);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
SpectralNorm<InputDataType, OutputDataType, CustomLayers...>::~SpectralNorm()
{
  delete wrappedLayer;
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
void SpectralNorm<InputDataType, OutputDataType, CustomLayers...>::Reset()
{
  // Set the weights of the inside layer to layerWeights.
  wrappedLayer->Parameters() = weights;
  wrappedLayer->Reset();
  weight = wrappedLayer->Weight();
  bias = wrappedLayer->Bias();
  v = arma::randu<arma::vec>(weight.n_cols);
  u = arma::randu<arma::vec>(weight.n_rows);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void SpectralNorm<InputDataType, OutputDataType, CustomLayers...>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  for (size_t i = 0; i < powerIterations; i++)
  {
    v = arma::normalise(weight.t() * u);
    u = arma::normalise(weight * v);
  }

  sigma = arma::as_scalar(u * (weight * v));
  weight /= sigma;
  wrappedLayer->Forward(std::move(input),
      std::move(output));
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void SpectralNorm<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  wrappedLayer->Backward(std::move(wrappedLayer->OutputParameter()),
      std::move(gy), std::move(g));
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void SpectralNorm<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  arma::mat output = weight * input;
  double lambda = 0;

  for (size_t i = 0; i < error.n_cols; i++){
    lambda += arma::as_scalar(error.col(i) * output.col(i));
  }

  gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
      error * input.t() -  (u * v.t()) * lambda);
  gradient.submat(weight.n_elem, 0, gradient.n_elem - 1, 0) =
      arma::sum(error, 1);

  gradient /= sigma;
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename Archive>
void SpectralNorm<InputDataType, OutputDataType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  if (Archive::is_loading::value)
  {
    delete wrappedLayer;
  }

  ar & BOOST_SERIALIZATION_NVP(wrappedLayer);
  ar & BOOST_SERIALIZATION_NVP(layerWeightSize);

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    weights.set_size(layerWeightSize, 1);
  }
}

} // namespace ann
} // namespace mlpack

#endif
