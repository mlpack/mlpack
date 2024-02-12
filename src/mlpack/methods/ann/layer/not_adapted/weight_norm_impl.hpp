/**
 * @file methods/ann/layer/weight_norm_impl.hpp
 * @author Toshal Agrawal
 *
 * Implementation of the WeightNorm Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_IMPL_HPP

// In case it is not included.
#include "weight_norm.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
WeightNormType<InputType, OutputType>::
WeightNormType() : wrappedLayer(new LinearType<InputType, OutputType>())
{
  layerWeightSize = wrappedLayer->WeightSize();
  weights.set_size(layerWeightSize + 1, 1);

  layerWeights.set_size(layerWeightSize, 1);
  layerGradients.set_size(layerWeightSize, 1);
}

template<typename InputType, typename OutputType>
WeightNormType<InputType, OutputType>::
WeightNormType(Layer<InputType, OutputType>* layer) : wrappedLayer(layer)
{
  layerWeightSize = wrappedLayer->WeightSize();
  weights.set_size(layerWeightSize + 1, 1);

  layerWeights.set_size(layerWeightSize, 1);
  layerGradients.set_size(layerWeightSize, 1);
}

template<typename InputType, typename OutputType>
WeightNormType<InputType, OutputType>::WeightNormType(
    const WeightNormType<InputType, OutputType>& other) :
    wrappedLayer(other.wrappedLayer->Clone()),
    layerWeightSize(other.layerWeightSize),
    weights(other.weights),
    layerGradients(other.layerGradients),
    layerWeights(other.layerWeights)
{
  // Nothing else to do.
}

template<typename InputType, typename OutputType>
WeightNormType<InputType, OutputType>::WeightNormType(
    WeightNormType<InputType, OutputType>&& other) :
    wrappedLayer(std::move(other.wrappedLayer)),
    layerWeightSize(other.layerWeightSize),
    weights(std::move(other.weights)),
    layerGradients(std::move(other.layerGradients)),
    layerWeights(std::move(other.layerWeights))
{
  // Reset the other layer.
  other = WeightNormType<InputType, OutputType>();
}

template<typename InputType, typename OutputType>
WeightNormType<InputType, OutputType>&
WeightNormType<InputType, OutputType>::operator=(
    const WeightNormType<InputType, OutputType>& other)
{
  if (this != &other)
  {
    wrappedLayer = other.wrappedLayer->Clone();
    layerWeightSize = other.layerWeightSize;
    weights = other.weights;
    layerWeights = other.layerWeights;
    layerGradients = other.layerGradients;
  }

  return *this;
}

template<typename InputType, typename OutputType>
WeightNormType<InputType, OutputType>&
WeightNormType<InputType, OutputType>::operator=(
    WeightNormType<InputType, OutputType>&& other)
{
  if (this != &other)
  {
    wrappedLayer = std::move(other.wrappedLayer);
    layerWeightSize = other.layerWeightSize;
    weights = std::move(other.weights);
    layerWeights = std::move(other.layerWeights);
    layerGradients = std::move(other.layerGradients);

    // Reset the other layer.
    other = WeightNormType<InputType, OutputType>();
  }

  return *this;
}

template<typename InputType, typename OutputType>
WeightNormType<InputType, OutputType>::~WeightNormType()
{
  delete wrappedLayer;
}

template<typename InputType, typename OutputType>
void WeightNormType<InputType, OutputType>::SetWeights(
    typename OutputType::elem_type* weightsPtr)
{
  // Set the weights of the inside layer to layerWeights.
  // This is done to set the non-bias terms correctly.
  /* boost::apply_visitor(WeightSetVisitor(layerWeights, 0), wrappedLayer); */
  wrappedLayer->SetWeights(weightsPtr);
  wrappedLayer->Parameters() = OutputType(layerWeights.memptr(),
      wrappedLayer->Parameters().n_rows, wrappedLayer->Parameters().n_cols,
      false, false);

  /* biasWeightSize = boost::apply_visitor(BiasSetVisitor(weights, 0), */
  /*     wrappedLayer); */
  biasWeightSize = 0;

  vectorParameter = OutputType(weights.memptr() + biasWeightSize,
      layerWeightSize - biasWeightSize, 1, false, false);

  scalarParameter = OutputType(weights.memptr() + layerWeightSize, 1, 1, false,
      false);
}

template<typename InputType, typename OutputType>
void WeightNormType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  // Initialize the non-bias weights of wrapped layer.
  const double normVectorParameter = norm(vectorParameter, 2);
  layerWeights.rows(0, layerWeightSize - biasWeightSize - 1) =
      scalarParameter(0) * vectorParameter / normVectorParameter;

  wrappedLayer->Forward(input, output);
}

template<typename InputType, typename OutputType>
void WeightNormType<InputType, OutputType>::Backward(
    const InputType& input, const OutputType& gy, OutputType& g)
{
  wrappedLayer->Backward(input, gy, g);
}

// TODO: this part is not trivial...
template<typename InputType, typename OutputType>
void WeightNormType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  ResetGradients(layerGradients);

  // Calculate the gradients of the wrapped layer.
  wrappedLayer->Gradient(input, error, gradient);

  // Store the norm of vector parameter temporarily.
  const double normVectorParameter = norm(vectorParameter, 2);

  // Set the gradients of the bias terms.
  if (biasWeightSize != 0)
  {
    gradient.rows(0, biasWeightSize - 1) = OutputType(layerGradients.memptr() +
        layerWeightSize - biasWeightSize, biasWeightSize, 1, false, false);
  }

  // Calculate the gradients of the scalar parameter.
  gradient[gradient.n_rows - 1] = accu(layerGradients.rows(0,
      layerWeightSize - biasWeightSize - 1) % vectorParameter) /
      normVectorParameter;

  // Calculate the gradients of the vector parameter.
  gradient.rows(biasWeightSize, layerWeightSize - 1) =
      scalarParameter(0) / normVectorParameter * (layerGradients.rows(0,
      layerWeightSize - biasWeightSize - 1) - gradient[gradient.n_rows - 1] /
      normVectorParameter * vectorParameter);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void WeightNormType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>*>(this));

  ar(CEREAL_POINTER(wrappedLayer));
  ar(CEREAL_NVP(layerWeightSize));
}

} // namespace mlpack

#endif
