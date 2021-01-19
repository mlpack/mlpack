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
namespace ann { /** Artificial Neural Network. */

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
WeightNormType<InputType, OutputType>::~WeightNormType()
{
  delete wrappedLayer;
}

template<typename InputType, typename OutputType>
void WeightNormType<InputType, OutputType>::Reset()
{
  // Set the weights of the inside layer to layerWeights.
  // This is done to set the non-bias terms correctly.
  /* boost::apply_visitor(WeightSetVisitor(layerWeights, 0), wrappedLayer); */
  wrappedLayer->Parameters() = OutputType(layerWeights.memptr(),
      wrappedLayer->Parameters().n_rows, wrappedLayer->Parameters().n_cols,
      false, false);

  /* boost::apply_visitor(ResetVisitor(), wrappedLayer); */
  wrappedLayer->Reset();

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
  const double normVectorParameter = arma::norm(vectorParameter, 2);
  layerWeights.rows(0, layerWeightSize - biasWeightSize - 1) =
      scalarParameter(0) * vectorParameter / normVectorParameter;

  wrappedLayer->Forward(input, wrappedLayer->OutputParameter());

  output = wrappedLayer->OutputParameter();
}

template<typename InputType, typename OutputType>
void WeightNormType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  wrappedLayer->Backward(
      wrappedLayer->OutputParameter(),
      gy,
      wrappedLayer->Delta()
  );

  g = wrappedLayer->Delta();
}

template<typename InputType, typename OutputType>
void WeightNormType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  ResetGradients(layerGradients);

  // Calculate the gradients of the wrapped layer.
  wrappedLayer->Gradient(
      input,
      error,
      wrappedLayer->Gradient()
  );

  // Store the norm of vector parameter temporarily.
  const double normVectorParameter = arma::norm(vectorParameter, 2);

  // Set the gradients of the bias terms.
  if (biasWeightSize != 0)
  {
    gradient.rows(0, biasWeightSize - 1) = OutputType(layerGradients.memptr() +
        layerWeightSize - biasWeightSize, biasWeightSize, 1, false, false);
  }

  // Calculate the gradients of the scalar parameter.
  gradient[gradient.n_rows - 1] = arma::accu(layerGradients.rows(0,
      layerWeightSize - biasWeightSize - 1) % vectorParameter) /
      normVectorParameter;

  // Calculate the gradients of the vector parameter.
  gradient.rows(biasWeightSize, layerWeightSize - 1) =
      scalarParameter(0) / normVectorParameter * (layerGradients.rows(0,
      layerWeightSize - biasWeightSize - 1) - gradient[gradient.n_rows - 1] /
      normVectorParameter * vectorParameter);
}

template<typename InputType, typename OutputType>
void WeightNormType<InputType, OutputType>::ResetGradients(OutputType& gradient)
{
  // boost::apply_visitor(GradientSetVisitor(gradient, 0), wrappedLayer);
  wrappedLayer->Gradient() = OutputType(gradient.memptr(),
      weights.n_rows, weights.n_cols, false, false);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void WeightNormType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  if (cereal::is_loading<Archive>())
  {
    delete wrappedLayer;
  }

  // ar(CEREAL_VARIANT_POINTER(wrappedLayer));
  ar(CEREAL_NVP(layerWeightSize));

  // If we are loading, we need to initialize the weights.
  if (cereal::is_loading<Archive>())
  {
    weights.set_size(layerWeightSize + 1, 1);
  }
}

} // namespace ann
} // namespace mlpack

#endif
