/**
 * @file inception_layer_impl.hpp
 * @author Marcus Edel
 * @author Toshal Agrawal
 *
 * Implementation of the InceptionLayer Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_IMPL_HPP

// In case it is not included.
#include "inception_layer.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
InceptionLayer<InputDataType, OutputDataType, CustomLayers...>::InceptionLayer()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
InceptionLayer<InputDataType, OutputDataType, CustomLayers...>::InceptionLayer(
    const size_t inSize,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t outOne,
    const size_t outTwo,
    const size_t outThree,
    const size_t outFour,
    const size_t outMidTwo,
    const size_t outMidThree)
{
  layer = new Concat<>(true);;

  Sequential<InputDataType, OutputDataType>* networkTwo, networkThree,
      networkFour;

  // Build the first network.
  Convolution<>* layerOne = new Convolution<>(inSize, outOne, 1, 1, 1, 1, 0, 0,
      inputWidth, inputHeight);
  layer->Add(layerOne);

  // Build the second network.
  Convolution<>* layerTwo = new Convolution<>(inSize, outMidTwo, 1, 1, 1, 1, 0,
      0, inputWidth, inputHeight);
  Convolution<>* layerThree = new Convolution<>(outMidTwo, outTwo, 3, 3, 1, 1,
      1, 1, inputWidth, inputHeight);
  networkTwo = new Sequential<>;
  networkTwo->Add(layerTwo);
  networkTwo->Add(layerThree);
  layer->Add(networkTwo);

  // Build the third network.
  Convolution<>* layerFour = new Convolution<>(inSize, outMidThree, 1, 1, 1, 1,
      0, 0, inputWidth, inputHeight);
  Convolution<>* layerFive = new Convolution<>(outMidThree, outThree, 5, 5, 1,
      1, 0, 0, inputWidth, inputHeight);
  networkThree = new Sequential<>;
  networkThree->Add(layerFour);
  networkThree->Add(layerFive);
  layer->Add(networkThree);

  // Build the fourth network.
  MaxPooling<>* layerSix = new MaxPooling<>(3, 3, 1, 1);
  Convolution<>* layerSeven = new Convolution<>(inSize, outFour, 1, 1, 1, 1, 0,
      0, inputWidth, inputHeight);
  networkFour = new Sequential<>;
  networkFour->Add(layerSix);
  networkFour->Add(layerSeven);
  layer->Add(networkFour);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
InceptionLayer<InputDataType, OutputDataType, CustomLayers...>::~InceptionLayer(
    )
{
  delete layer;
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void InceptionLayer<InputDataType, OutputDataType, CustomLayers...>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  layer->Forward(std::move(input), std::move(output));
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void InceptionLayer<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>&&  input, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  layer->Backward(std::move(input), std::move(gy), std::move(g));
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void InceptionLayer<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  layer->Gradient(std::move(input), std::move(error), std::move(gradient));
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename Archive>
void InceptionLayer<InputDataType, OutputDataType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  if (Archive::is_loading::value)
  {
    delete layer;
  }

  ar & BOOST_SERIALIZATION_NVP(layer);
}

} // namespace ann
} // namespace mlpack

#endif
