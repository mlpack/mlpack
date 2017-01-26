/**
 * @file layer_visitor_impl.hpp
 * @author Marcus Edel
 *
 * This file provides an easy way to serialize a layer, abstracts away the
 * different types of layers, and also automatically directs any function to the
 * right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYER_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYER_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "layer_visitor.hpp"

namespace mlpack {
namespace ann {

//! LoadOutputParameterVisitor visitor class.
inline LoadOutputParameterVisitor::LoadOutputParameterVisitor(
    std::vector<arma::mat>&& parameter) : parameter(std::move(parameter))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void LoadOutputParameterVisitor::operator()(LayerType* layer) const
{
  OutputParameter(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
LoadOutputParameterVisitor::OutputParameter(T* layer) const
{
  layer->OutputParameter() = parameter.back();
  parameter.pop_back();
}

template<typename T>
inline typename std::enable_if<
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
LoadOutputParameterVisitor::OutputParameter(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(LoadOutputParameterVisitor(std::move(parameter)),
        layer->Model()[layer->Model().size() - i - 1]);
  }

  layer->OutputParameter() = parameter.back();
  parameter.pop_back();
}

//! SaveOutputParameterVisitor visitor class.
inline SaveOutputParameterVisitor::SaveOutputParameterVisitor(
    std::vector<arma::mat>&& parameter) : parameter(std::move(parameter))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void SaveOutputParameterVisitor::operator()(LayerType* layer) const
{
  OutputParameter(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
SaveOutputParameterVisitor::OutputParameter(T* layer) const
{
  parameter.push_back(layer->OutputParameter());
}

template<typename T>
inline typename std::enable_if<
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
SaveOutputParameterVisitor::OutputParameter(T* layer) const
{
  parameter.push_back(layer->OutputParameter());

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SaveOutputParameterVisitor(std::move(parameter)),
        layer->Model()[i]);
  }
}

//! DeleteVisitor visitor class.
template<typename LayerType>
inline void DeleteVisitor::operator()(LayerType* layer) const
{
  if (layer)
    delete layer;
}

//! ForwardOutputVisitor visitor class.
inline ForwardOutputVisitor::ForwardOutputVisitor(arma::mat&& input,
                                                  arma::mat&& output) :
  input(std::move(input)),
  output(std::move(output))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline double ForwardOutputVisitor::operator()(LayerType* layer) const
{
  return layer->Forward(std::move(input), std::move(output));
}

//! ForwardVisitor visitor class.
inline ForwardVisitor::ForwardVisitor(arma::mat&& input, arma::mat&& output) :
  input(std::move(input)),
  output(std::move(output))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void ForwardVisitor::operator()(LayerType* layer) const
{
  layer->Forward(std::move(input), std::move(output));
}

//! BackwardVisitor visitor class.
inline BackwardVisitor::BackwardVisitor(arma::mat&& input,
                                 arma::mat&& error,
                                 arma::mat&& delta) :
  input(std::move(input)),
  error(std::move(error)),
  delta(std::move(delta))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void BackwardVisitor::operator()(LayerType* layer) const
{
  layer->Backward(std::move(input), std::move(error), std::move(delta));
}

//! ResetVisitor visitor class.
template<typename LayerType>
inline void ResetVisitor::operator()(LayerType* layer) const
{
  ResetParameter(layer);
}

template<typename T>
inline typename std::enable_if<
    HasResetCheck<T, void(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
ResetVisitor::ResetParameter(T* layer) const
{
  layer->Reset();
}

template<typename T>
inline typename std::enable_if<
    !HasResetCheck<T, void(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
ResetVisitor::ResetParameter(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(ResetVisitor(), layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    HasResetCheck<T, void(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
ResetVisitor::ResetParameter(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(ResetVisitor(), layer->Model()[i]);
  }

  layer->Reset();
}

template<typename T>
inline typename std::enable_if<
    !HasResetCheck<T, void(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
ResetVisitor::ResetParameter(T* /* layer */) const
{
  /* Nothing to do here. */
}

//! InputParameterVisitor visitor class.
template<typename LayerType>
inline arma::mat& InputParameterVisitor::operator()(LayerType *layer) const
{
  return layer->InputParameter();
}

//! OutputParameterVisitor visitor class.
template<typename LayerType>
inline arma::mat& OutputParameterVisitor::operator()(LayerType *layer) const
{
  return layer->OutputParameter();
}

//! DeltaVisitor visitor class.
template<typename LayerType>
inline arma::mat& DeltaVisitor::operator()(LayerType *layer) const
{
  return layer->Delta();
}

//! ParametersVisitor visitor class.
inline ParametersVisitor::ParametersVisitor(arma::mat&& parameters) :
    parameters(std::move(parameters))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void ParametersVisitor::operator()(LayerType *layer) const
{
  LayerParameters(layer, layer->OutputParameter());
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value, void>::type
ParametersVisitor::LayerParameters(T* /* layer */, P& /* output */) const
{
  /* Nothing to do here. */
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value, void>::type
ParametersVisitor::LayerParameters(T* layer, P& /* output */) const
{
  parameters = layer->Parameters();
}

//! ParametersSetVisitor visitor class.
inline ParametersSetVisitor::ParametersSetVisitor(arma::mat&& parameters) :
    parameters(std::move(parameters))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void ParametersSetVisitor::operator()(LayerType *layer) const
{
  LayerParameters(layer, layer->OutputParameter());
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value, void>::type
ParametersSetVisitor::LayerParameters(T* /* layer */, P& /* output */) const
{
  /* Nothing to do here. */
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value, void>::type
ParametersSetVisitor::LayerParameters(T* layer, P& /* output */) const
{
  layer->Parameters() = parameters;
}

//! WeightSizeVisitor visitor class.
template<typename LayerType>
inline size_t WeightSizeVisitor::operator()(LayerType* layer) const
{
  return LayerSize(layer, layer->OutputParameter());
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSizeVisitor::LayerSize(T* /* layer */, P& /* output */) const
{
  return 0;
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSizeVisitor::LayerSize(T* layer, P& /* output */) const
{
  size_t weights = 0;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    weights += boost::apply_visitor(WeightSizeVisitor(), layer->Model()[i]);
  }

  return weights;
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSizeVisitor::LayerSize(T* layer, P& /* output */) const
{
  return layer->Parameters().n_elem;
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSizeVisitor::LayerSize(T* layer, P& /* output */) const
{
  size_t weights = layer->Parameters().n_elem;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    weights += boost::apply_visitor(WeightSizeVisitor(), layer->Model()[i]);
  }

  return weights;
}

//! SetInputWidthVisitor visitor class.
inline SetInputWidthVisitor::SetInputWidthVisitor(const size_t inputWidth,
                                                  const bool reset) :
    inputWidth(inputWidth),
    reset(reset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline bool SetInputWidthVisitor::operator()(LayerType* layer) const
{
  return LayerInputWidth(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputWidthVisitor::LayerInputWidth(T* /* layer */) const
{
  return false;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputWidthVisitor::LayerInputWidth(T* layer) const
{
  if (layer->InputWidth() == 0 || reset)
  {
    layer->InputWidth() = inputWidth;
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputWidthVisitor::LayerInputWidth(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputWidthVisitor(inputWidth, reset),
        layer->Model()[i]);
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputWidthVisitor::LayerInputWidth(T* layer) const
{
  if (layer->InputWidth() == 0 || reset)
  {
    layer->InputWidth() = inputWidth;
  }

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputWidthVisitor(inputWidth, reset),
        layer->Model()[i]);
  }

  return true;
}

//! SetInputHeightVisitor visitor class.
inline SetInputHeightVisitor::SetInputHeightVisitor(const size_t inputHeight,
                                                    const bool reset) :
    inputHeight(inputHeight),
    reset(reset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline bool SetInputHeightVisitor::operator()(LayerType* layer) const
{
  return LayerInputHeight(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputHeightVisitor::LayerInputHeight(T* /* layer */) const
{
  return false;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputHeightVisitor::LayerInputHeight(T* layer) const
{
  if (layer->InputHeight() == 0 || reset)
  {
    layer->InputHeight() = inputHeight;
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputHeightVisitor::LayerInputHeight(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputHeightVisitor(inputHeight, reset),
        layer->Model()[i]);
  }

  return true;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
SetInputHeightVisitor::LayerInputHeight(T* layer) const
{
  if (layer->InputHeight() == 0  || reset)
  {
    layer->InputHeight() = inputHeight;
  }

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SetInputHeightVisitor(inputHeight, reset),
        layer->Model()[i]);
  }

  return true;
}

//! OutputWidthVisitor visitor class.
template<typename LayerType>
inline size_t OutputWidthVisitor::operator()(LayerType* layer) const
{
  return LayerOutputWidth(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputWidthVisitor::LayerOutputWidth(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputWidthVisitor::LayerOutputWidth(T* layer) const
{
  return layer->OutputWidth();
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputWidthVisitor::LayerOutputWidth(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    size_t outputWidth = boost::apply_visitor(OutputWidthVisitor(),
        layer->Model()[layer->Model().size() - 1 - i]);

    if (outputWidth != 0)
    {
      return outputWidth;
    }
  }

  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputWidthVisitor::LayerOutputWidth(T* layer) const
{
  size_t outputWidth = layer->OutputWidth();

  if (outputWidth == 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      outputWidth = boost::apply_visitor(OutputWidthVisitor(),
          layer->Model()[layer->Model().size() - 1 - i]);

      if (outputWidth != 0)
      {
        return outputWidth;
      }
    }
  }

  return outputWidth;
}

//! OutputHeightVisitor visitor class.
template<typename LayerType>
inline size_t OutputHeightVisitor::operator()(LayerType* layer) const
{
  return LayerOutputHeight(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputHeightVisitor::LayerOutputHeight(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputHeightVisitor::LayerOutputHeight(T* layer) const
{
  return layer->OutputHeight();
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputHeightVisitor::LayerOutputHeight(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    size_t outputHeight = boost::apply_visitor(OutputHeightVisitor(),
        layer->Model()[layer->Model().size() - 1 - i]);

    if (outputHeight != 0)
    {
      return outputHeight;
    }
  }

  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
OutputHeightVisitor::LayerOutputHeight(T* layer) const
{
  size_t outputHeight = layer->OutputHeight();

  if (outputHeight == 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      outputHeight = boost::apply_visitor(OutputHeightVisitor(),
          layer->Model()[layer->Model().size() - 1 - i]);

      if (outputHeight != 0)
      {
        return outputHeight;
      }
    }
  }

  return outputHeight;
}

//! LastOutputWidthVisitor visitor class.
template<typename LayerType>
inline size_t LastOutputWidthVisitor::operator()(LayerType* layer) const
{
  return LayerOutputWidth(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
LastOutputWidthVisitor::LayerOutputWidth(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
LastOutputWidthVisitor::LayerOutputWidth(T* layer) const
{
  return layer->OutputWidth();
}

template<typename T>
inline typename std::enable_if<
    !HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
LastOutputWidthVisitor::LayerOutputWidth(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    size_t outputWidth = boost::apply_visitor(LastOutputWidthVisitor(),
        layer->Model()[layer->Model().size() - 1 - i]);

    if (outputWidth != 0)
    {
      return outputWidth;
    }
  }

  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputWidth<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
LastOutputWidthVisitor::LayerOutputWidth(T* layer) const
{
  size_t outputWidth = layer->OutputWidth();

  if (outputWidth == 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      outputWidth = boost::apply_visitor(OutputWidthVisitor(),
          layer->Model()[layer->Model().size() - 1 - i]);

      if (outputWidth != 0)
      {
        return outputWidth;
      }
    }
  }

  return outputWidth;
}

//! LastOutputHeightVisitor visitor class.
template<typename LayerType>
inline size_t LastOutputHeightVisitor::operator()(LayerType* layer) const
{
  return LayerOutputHeight(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
LastOutputHeightVisitor::LayerOutputHeight(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
LastOutputHeightVisitor::LayerOutputHeight(T* layer) const
{
  return layer->OutputHeight();
}

template<typename T>
inline typename std::enable_if<
    !HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
LastOutputHeightVisitor::LayerOutputHeight(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    size_t outputHeight = boost::apply_visitor(LastOutputHeightVisitor(),
        layer->Model()[layer->Model().size() - 1 - i]);

    if (outputHeight != 0)
    {
      return outputHeight;
    }
  }

  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasInputHeight<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
LastOutputHeightVisitor::LayerOutputHeight(T* layer) const
{
  size_t outputHeight = layer->OutputHeight();

  if (outputHeight == 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      outputHeight = boost::apply_visitor(OutputHeightVisitor(),
          layer->Model()[layer->Model().size() - 1 - i]);

      if (outputHeight != 0)
      {
        return outputHeight;
      }
    }
  }

  return outputHeight;
}

//! WeightSetVisitor visitor class.
inline WeightSetVisitor::WeightSetVisitor(arma::mat&& weight,
                                          const size_t offset) :
    weight(std::move(weight)),
    offset(offset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline size_t WeightSetVisitor::operator()(LayerType* layer) const
{
  return LayerSize(layer, std::move(layer->OutputParameter()));
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSetVisitor::LayerSize(T* /* layer */, P&& /*output */) const
{
  return 0;
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSetVisitor::LayerSize(T* layer, P&& /*output */) const
{
  size_t modelOffset = 0;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(WeightSetVisitor(
        std::move(weight), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSetVisitor::LayerSize(T* layer, P&& /* output */) const
{
  layer->Parameters() = arma::mat(weight.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

  return layer->Parameters().n_elem;
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
WeightSetVisitor::LayerSize(T* layer, P&& /* output */) const
{
  layer->Parameters() = arma::mat(weight.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

  size_t modelOffset = layer->Parameters().n_elem;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(WeightSetVisitor(
        std::move(weight), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

//! RhoVisitor visitor class.
template<typename LayerType>
inline size_t RhoVisitor::operator()(LayerType* layer) const
{
  return LayerRho(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasRho<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
RhoVisitor::LayerRho(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    !HasRho<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
RhoVisitor::LayerRho(T* layer) const
{
  size_t moduleRho = 0;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    moduleRho = boost::apply_visitor(RhoVisitor(), layer->Model()[i]);
    if (moduleRho != 0)
    {
      return moduleRho;
    }
  }

  return moduleRho;
}

template<typename T>
inline typename std::enable_if<
    HasRho<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
RhoVisitor::LayerRho(T* layer) const
{
  return layer->Rho();
}

template<typename T>
inline typename std::enable_if<
    HasRho<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
RhoVisitor::LayerRho(T* layer) const
{
  return layer->Rho();
}

//! DeterministicSetVisitor visitor class.
inline DeterministicSetVisitor::DeterministicSetVisitor(
    const bool deterministic) : deterministic(deterministic)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void DeterministicSetVisitor::operator()(LayerType* layer) const
{
  LayerDeterministic(layer);
}

template<typename T>
inline typename std::enable_if<
    HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
DeterministicSetVisitor::LayerDeterministic(T* layer) const
{
  layer->Deterministic() = deterministic;

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(DeterministicSetVisitor(deterministic),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
DeterministicSetVisitor::LayerDeterministic(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(DeterministicSetVisitor(deterministic),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
DeterministicSetVisitor::LayerDeterministic(T* layer) const
{
  layer->Deterministic() = deterministic;
}

template<typename T>
inline typename std::enable_if<
    !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
DeterministicSetVisitor::LayerDeterministic(T* /* input */) const
{
  /* Nothing to do here. */
}

//! AddVisitor visitor class.
template<typename T>
inline AddVisitor::AddVisitor(T newLayer) :
    newLayer(std::move(newLayer))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void AddVisitor::operator()(LayerType* layer) const
{
  LayerAdd<LayerType>(layer);
}

template<typename T>
inline typename std::enable_if<
    HasAddCheck<T, void(T::*)(LayerTypes)>::value, void>::type
AddVisitor::LayerAdd(T* layer) const
{
  layer->Add(newLayer);
}

template<typename T>
inline typename std::enable_if<
    !HasAddCheck<T, void(T::*)(LayerTypes)>::value, void>::type
AddVisitor::LayerAdd(T* /* layer */) const
{
  /* Nothing to do here. */
}

//! GradientSetVisitor visitor class.
inline GradientSetVisitor::GradientSetVisitor(arma::mat&& gradient,
                                              size_t offset) :
    gradient(std::move(gradient)),
    offset(offset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline size_t GradientSetVisitor::operator()(LayerType* layer) const
{
  return LayerGradients(layer, layer->OutputParameter());
}

template<typename T>
inline typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientSetVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  layer->Gradient() = arma::mat(gradient.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

  return layer->Parameters().n_elem;
}

template<typename T>
inline typename std::enable_if<
    !HasGradientCheck<T, arma::mat&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientSetVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  size_t modelOffset = 0;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(GradientSetVisitor(
        std::move(gradient), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

template<typename T>
inline typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientSetVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  layer->Gradient() = arma::mat(gradient.memptr() + offset,
      layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);

  size_t modelOffset = layer->Parameters().n_elem;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(GradientSetVisitor(
        std::move(gradient), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasGradientCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientSetVisitor::LayerGradients(T* /* layer */, P& /* input */) const
{
  return 0;
}

//! GradientUpdateVisitor visitor class.
inline GradientUpdateVisitor::GradientUpdateVisitor(arma::mat&& gradient,
                                                    size_t offset) :
    gradient(std::move(gradient)),
    offset(offset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline size_t GradientUpdateVisitor::operator()(LayerType* layer) const
{
  return LayerGradients(layer, layer->OutputParameter());
}

template<typename T>
inline typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientUpdateVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  if (layer->Parameters().n_elem != 0)
  {
    layer->Gradient() = gradient.submat(offset, 0,
        offset + layer->Parameters().n_elem - 1, 0);;
  }

  return layer->Parameters().n_elem;
}

template<typename T>
inline typename std::enable_if<
    !HasGradientCheck<T, arma::mat&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientUpdateVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  size_t modelOffset = 0;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(GradientUpdateVisitor(
        std::move(gradient), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

template<typename T>
inline typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientUpdateVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  if (layer->Parameters().n_elem != 0)
  {
    layer->Gradient() = gradient.submat(offset, 0,
        offset + layer->Parameters().n_elem - 1, 0);;
  }

  size_t modelOffset = layer->Parameters().n_elem;
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(GradientUpdateVisitor(
        std::move(gradient), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasGradientCheck<T, P&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, size_t>::type
GradientUpdateVisitor::LayerGradients(T* /* layer */, P& /* input */) const
{
  return 0;
}

//! GradientZeroVisitor visitor class.
inline GradientZeroVisitor::GradientZeroVisitor()
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void GradientZeroVisitor::operator()(LayerType* layer) const
{
  LayerGradients(layer, layer->OutputParameter());
}

template<typename T>
inline typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value, void>::type
GradientZeroVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  layer->Gradient().zeros();
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasGradientCheck<T, P&(T::*)()>::value, void>::type
GradientZeroVisitor::LayerGradients(T* /* layer */, P& /* input */) const
{
  /* Nothing to do here. */
}

//! GradientVisitor visitor class.
inline GradientVisitor::GradientVisitor(arma::mat&& input, arma::mat&& delta) :
    input(std::move(input)),
    delta(std::move(delta))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void GradientVisitor::operator()(LayerType* layer) const
{
  LayerGradients(layer, layer->OutputParameter());
}

template<typename T>
inline typename std::enable_if<
    HasGradientCheck<T, arma::mat&(T::*)()>::value, void>::type
GradientVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
{
  layer->Gradient(std::move(input), std::move(delta),
      std::move(layer->Gradient()));
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasGradientCheck<T, P&(T::*)()>::value, void>::type
GradientVisitor::LayerGradients(T* /* layer */, P& /* input */) const
{
  /* Nothing to do here. */
}

//! RewardSetVisitor visitor class.
inline RewardSetVisitor::RewardSetVisitor(const double reward) : reward(reward)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void RewardSetVisitor::operator()(LayerType* layer) const
{
  LayerReward(layer);
}

template<typename T>
inline typename std::enable_if<
    HasRewardCheck<T, double&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
RewardSetVisitor::LayerReward(T* layer) const
{
  layer->Reward() = reward;

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(RewardSetVisitor(reward),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    !HasRewardCheck<T, double&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
RewardSetVisitor::LayerReward(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(RewardSetVisitor(reward),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    HasRewardCheck<T, double&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
RewardSetVisitor::LayerReward(T* layer) const
{
  layer->Reward() = reward;
}

template<typename T>
inline typename std::enable_if<
    !HasRewardCheck<T, double&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
RewardSetVisitor::LayerReward(T* /* input */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
