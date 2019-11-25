/**
 * @file weight_norm_impl.hpp
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

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"
#include "../visitor/bias_set_visitor.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
WeightNorm<InputDataType, OutputDataType, CustomLayers...>::WeightNorm(
    LayerTypes<CustomLayers...> layer) :
    wrappedLayer(layer)
{
  layerWeightSize = boost::apply_visitor(weightSizeVisitor, wrappedLayer);
  weights.set_size(layerWeightSize + 1, 1);

  layerWeights.set_size(layerWeightSize, 1);
  layerGradients.set_size(layerWeightSize, 1);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
WeightNorm<InputDataType, OutputDataType, CustomLayers...>::~WeightNorm()
{
  boost::apply_visitor(deleteVisitor, wrappedLayer);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Reset()
{
  // Set the weights of the inside layer to layerWeights.
  // This is done to set the non-bias terms correctly.
  boost::apply_visitor(WeightSetVisitor(std::move(layerWeights), 0),
      wrappedLayer);

  boost::apply_visitor(resetVisitor, wrappedLayer);

  biasWeightSize = boost::apply_visitor(BiasSetVisitor(std::move(weights),
      0), wrappedLayer);

  vectorParameter = arma::mat(weights.memptr() + biasWeightSize,
      layerWeightSize - biasWeightSize, 1, false, false);

  scalarParameter = arma::mat(weights.memptr() + layerWeightSize, 1, 1, false,
      false);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  // Initialize the non-bias weights of wrapped layer.
  const double normVectorParameter = arma::norm(vectorParameter, 2);
  layerWeights.rows(0, layerWeightSize - biasWeightSize - 1) =
      scalarParameter(0) * vectorParameter / normVectorParameter;

  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, wrappedLayer))),
      wrappedLayer);

  output = boost::apply_visitor(outputParameterVisitor, wrappedLayer);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, wrappedLayer)), std::move(gy), std::move(
      boost::apply_visitor(deltaVisitor, wrappedLayer))), wrappedLayer);

  g = boost::apply_visitor(deltaVisitor, wrappedLayer);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  ResetGradients(layerGradients);

  // Calculate the gradients of the wrapped layer.
  boost::apply_visitor(GradientVisitor(std::move(input),
      std::move(error)), wrappedLayer);

  // Store the norm of vector parameter temporarily.
  const double normVectorParameter = arma::norm(vectorParameter, 2);

  // Set the gradients of the bias terms.
  if (biasWeightSize != 0)
  {
    gradient.rows(0, biasWeightSize - 1) = arma::mat(layerGradients.memptr() +
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

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::ResetGradients(
    arma::mat& gradient)
{
  boost::apply_visitor(GradientSetVisitor(std::move(gradient), 0),
      wrappedLayer);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename Archive>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  if (Archive::is_loading::value)
  {
    boost::apply_visitor(deleteVisitor, wrappedLayer);
  }

  ar & BOOST_SERIALIZATION_NVP(wrappedLayer);
  ar & BOOST_SERIALIZATION_NVP(layerWeightSize);

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    weights.set_size(layerWeightSize + 1, 1);
  }
}

} // namespace ann
} // namespace mlpack

#endif
