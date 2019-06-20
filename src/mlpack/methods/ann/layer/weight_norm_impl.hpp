/**
 * @file weight_norm_impl.hpp
 * @author Toshal Agrawal
 *
 * Implementation of the Weight Normalization Layer.
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

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
WeightNorm<InputDataType, OutputDataType, CustomLayers...>::WeightNorm(
    LayerTypes<CustomLayers...> layer) :
    wrappedLayer(layer)
{
  layerWeightSize = boost::apply_visitor(weightSizeVisitor, wrappedLayer);
  weights.set_size(2 * layerWeightSize + 1, 1);
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
  size_t offset = boost::apply_visitor(WeightSetVisitor(std::move(weights),
      0), wrappedLayer);
  boost::apply_visitor(resetVisitor, wrappedLayer);

  vectorParameter = arma::mat(weights.memptr() + offset, offset, 1, false,
      false);

  scalarParameter = arma::mat(weights.memptr() + 2 * offset, 1, 1, false,
      false);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  // Intialize the weights of wrapped layer.
  double normVectorParameter = arma::norm(vectorParameter, 2);
  weights.rows(0, layerWeightSize - 1) = scalarParameter(0) * vectorParameter
      / normVectorParameter;

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
  ResetGradients(gradient);

  // Calculate the gradients of the wrapped layer.
  boost::apply_visitor(GradientVisitor(std::move(input),
      std::move(error)), wrappedLayer);

  // Store the norm of vector parameter temporarily.
  double normVectorParameter = arma::norm(vectorParameter, 2);

  // Calculate the gradients of the scalar parameter.
  gradient[gradient.n_rows - 1] = arma::accu(gradient.rows(0, layerWeightSize
      - 1) % vectorParameter) / normVectorParameter;

  // Calculate the gradients of the vector parameter.
  gradient.rows(layerWeightSize, 2 * layerWeightSize - 1) =
      scalarParameter(0) / normVectorParameter * (gradient.rows(0,
      layerWeightSize - 1) - gradient[gradient.n_rows - 1] /
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
    weights.set_size(2 * layerWeightSize + 1, 1);
  }
}

} // namespace ann
} // namespace mlpack

#endif
