/**
 * @file methods/ann/layer/concat_impl.hpp
 * @author Marcus Edel
 * @author Mehul Kumar Nirala
 *
 * Implementation of the Concat class, which acts as a concatenation contain.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_IMPL_HPP

// In case it hasn't yet been included.
#include "concat.hpp"

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
Concat<InputDataType, OutputDataType, CustomLayers...>::Concat(
    const bool model, const bool run) :
    axis(0),
    useAxis(false),
    model(model),
    run(run),
    channels(1)
{
  parameters.set_size(0, 0);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
Concat<InputDataType, OutputDataType, CustomLayers...>::Concat(
    arma::Row<size_t>& inputSize,
    const size_t axis,
    const bool model,
    const bool run) :
    inputSize(inputSize),
    axis(axis),
    useAxis(true),
    model(model),
    run(run)
{
  parameters.set_size(0, 0);

  // Parameters to help calculate the number of channels.
  size_t oldColSize = 1, newColSize = 1;
  // Axis is specified and useAxis is true.
  if (useAxis)
  {
    // Axis is specified without input dimension.
    // Throw an error.
    if (inputSize.n_elem > 0)
    {
      // Calculate rowSize, newColSize based on the axis
      // of concatenation. Finally concat along cols and
      // reshape to original format i.e. (input, batch_size).
      size_t i = std::min(axis + 1, (size_t) inputSize.n_elem);
      for (; i < inputSize.n_elem; ++i)
      {
        newColSize *= inputSize[i];
      }
    }
    else
    {
      throw std::logic_error("Input dimensions not specified.");
    }
  }
  else
  {
    channels = 1;
  }
  if (newColSize <= 0)
  {
      throw std::logic_error("Col size is zero.");
  }
  channels = newColSize / oldColSize;
  inputSize.clear();
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
Concat<InputDataType, OutputDataType, CustomLayers...>::~Concat()
{
  if (!model)
  {
    // Clear memory.
    std::for_each(network.begin(), network.end(),
        boost::apply_visitor(deleteVisitor));
  }
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Concat<InputDataType, OutputDataType, CustomLayers...>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      boost::apply_visitor(ForwardVisitor(input,
          boost::apply_visitor(outputParameterVisitor, network[i])),
          network[i]);
    }
  }

  output = boost::apply_visitor(outputParameterVisitor, network.front());

  // Reshape output to incorporate the channels.
  output.reshape(output.n_rows / channels, output.n_cols * channels);

  for (size_t i = 1; i < network.size(); ++i)
  {
    arma::Mat<eT> out = boost::apply_visitor(outputParameterVisitor,
        network[i]);

    out.reshape(out.n_rows / channels, out.n_cols * channels);

    // Vertically concatentate output from each layer.
    output = arma::join_cols(output, out);
  }
  // Reshape output to its original shape.
  output.reshape(output.n_rows * channels, output.n_cols / channels);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Concat<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  size_t rowCount = 0;
  if (run)
  {
    arma::Mat<eT> delta;
    arma::Mat<eT> gyTmp(((arma::Mat<eT>&) gy).memptr(), gy.n_rows / channels,
        gy.n_cols * channels, false, false);
    for (size_t i = 0; i < network.size(); ++i)
    {
      // Use rows from the error corresponding to the output from each layer.
      size_t rows = boost::apply_visitor(
          outputParameterVisitor, network[i]).n_rows;

      // Extract from gy the parameters for the i-th network.
      delta = gyTmp.rows(rowCount / channels, (rowCount + rows) / channels - 1);
      delta.reshape(delta.n_rows * channels, delta.n_cols / channels);

      boost::apply_visitor(BackwardVisitor(
          boost::apply_visitor(outputParameterVisitor,
          network[i]), delta,
          boost::apply_visitor(deltaVisitor, network[i])), network[i]);
      rowCount += rows;
    }

    g = boost::apply_visitor(deltaVisitor, network[0]);
    for (size_t i = 1; i < network.size(); ++i)
    {
      g += boost::apply_visitor(deltaVisitor, network[i]);
    }
  }
  else
  {
    g = gy;
  }
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Concat<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g,
    const size_t index)
{
  size_t rowCount = 0, rows = 0;

  for (size_t i = 0; i < index; ++i)
  {
    rowCount += boost::apply_visitor(
        outputParameterVisitor, network[i]).n_rows;
  }
  rows = boost::apply_visitor(outputParameterVisitor, network[index]).n_rows;

  // Reshape gy to extract the i-th layer gy.
  arma::Mat<eT> gyTmp(((arma::Mat<eT>&) gy).memptr(), gy.n_rows / channels,
      gy.n_cols * channels, false, false);

  arma::Mat<eT> delta = gyTmp.rows(rowCount / channels, (rowCount + rows) /
      channels - 1);
  delta.reshape(delta.n_rows * channels, delta.n_cols / channels);

  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, network[index]), delta,
      boost::apply_visitor(deltaVisitor, network[index])), network[index]);

  g = boost::apply_visitor(deltaVisitor, network[index]);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Concat<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& /* gradient */)
{
  if (run)
  {
    size_t rowCount = 0;
    // Reshape error to extract the i-th layer error.
    arma::Mat<eT> errorTmp(((arma::Mat<eT>&) error).memptr(),
        error.n_rows / channels, error.n_cols * channels, false, false);
    for (size_t i = 0; i < network.size(); ++i)
    {
      size_t rows = boost::apply_visitor(
          outputParameterVisitor, network[i]).n_rows;

      // Extract from error the parameters for the i-th network.
      arma::Mat<eT> err = errorTmp.rows(rowCount / channels, (rowCount + rows) /
          channels - 1);
      err.reshape(err.n_rows * channels, err.n_cols / channels);

      boost::apply_visitor(GradientVisitor(input, err), network[i]);
      rowCount += rows;
    }
  }
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Concat<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& /* gradient */,
    const size_t index)
{
  size_t rowCount = 0;
  for (size_t i = 0; i < index; ++i)
  {
    rowCount += boost::apply_visitor(outputParameterVisitor,
        network[i]).n_rows;
  }
  size_t rows = boost::apply_visitor(
      outputParameterVisitor, network[index]).n_rows;

  arma::Mat<eT> errorTmp(((arma::Mat<eT>&) error).memptr(),
      error.n_rows / channels, error.n_cols * channels, false, false);
  arma::Mat<eT> err = errorTmp.rows(rowCount / channels, (rowCount + rows) /
      channels - 1);
  err.reshape(err.n_rows * channels, err.n_cols / channels);

  boost::apply_visitor(GradientVisitor(input, err), network[index]);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename Archive>
void Concat<InputDataType, OutputDataType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(model);
  ar & BOOST_SERIALIZATION_NVP(run);

  // Do we have to load or save a model?
  if (model)
  {
    // Clear memory first, if needed.
    if (Archive::is_loading::value)
    {
      std::for_each(network.begin(), network.end(),
          boost::apply_visitor(deleteVisitor));
    }

    ar & BOOST_SERIALIZATION_NVP(network);
  }
}

} // namespace ann
} // namespace mlpack


#endif
