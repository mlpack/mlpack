/**
 * @file concat_impl.hpp
 * @author Marcus Edel
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
    const bool model, const bool run) : model(model), run(run)
{
  parameters.set_size(0, 0);
  axis = -1;
  channels = 1;
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
Concat<InputDataType,
       OutputDataType,
       CustomLayers...
      >::Concat(
      arma::Row<arma::sword> inputSize,
      const int axis,
      const bool model,
      const bool run) :
      inputSize(inputSize),
      axis(axis),
      model(model),
      run(run)
{
  parameters.set_size(0, 0);
  int unknown = 0;
  if (axis < 0)
  {
    for (int i = 0; i < inputSize.n_elem; ++i)
    {
      if (inputSize[i] < 0)
      {
        Concat::axis = i;
        unknown++;
      }
    }
    if (unknown > 1)
    {
      throw std::logic_error("More than one dimension unknown.");
    }
  }
  Concat::channels = 1;
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
Concat<InputDataType, OutputDataType, CustomLayers...>::~Concat()
{
  // Clear memory.
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deleteVisitor));
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Concat<InputDataType, OutputDataType, CustomLayers...>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  if (run)
  {
    for (size_t i = 0; i < network.size(); ++i)
    {
      boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
          boost::apply_visitor(outputParameterVisitor, network[i]))),
          network[i]);
    }
  }

  // Parameter to store dimensions(rowSize).
  int rowSize, oldColSize, newColSize;
  output = boost::apply_visitor(outputParameterVisitor, network.front());

  newColSize = oldColSize = output.n_cols;

  // Axis is not specified.
  if (axis >= 0)
  {
    // Axis is specified without input dimension.
    // Throw an error.
    if (inputSize.n_elem > 0)
    {
      // Calculate rowSize, newColSize based on the axis
      // of concatenation. Finally concat along cols and
      // reshape to original format i.e. (input, batch_size).
      int i = std::min(axis + 1, (int) inputSize.n_elem);
      for (; i < inputSize.n_elem; ++i)
      {
        newColSize *= inputSize[i];
      }
    }
    else
    {
      throw std::logic_error("Input Dimensions not specified.");
    }
  }
  if (newColSize <= 0)
  {
      throw std::logic_error("Col Size is zero.");
  }

  channels = newColSize / oldColSize;
  // Compute the rowSize after which join_cols() is called.
  rowSize = output.n_rows * output.n_cols / newColSize;
  output.reshape(rowSize, newColSize);

  for (size_t i = 1; i < network.size(); ++i)
  {
    arma::Mat<eT> out = boost::apply_visitor(outputParameterVisitor,
        network[i]);

    rowSize = out.n_rows * out.n_cols / newColSize;
    out.reshape(rowSize, newColSize);

    // Vertically concatentate output from each layer.
    output = arma::join_cols(output, out);
  }

  rowSize = output.n_rows * output.n_cols / oldColSize;
  output.reshape(rowSize, oldColSize);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Concat<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  size_t rowCount = 0;
  if (run)
  {
    arma::Mat<eT> delta;
    gy.reshape(gy.n_rows / channels, gy.n_cols * channels);
    for (size_t i = 0; i < network.size(); ++i)
    {
      // Use rows from the error corresponding to the output from each layer.
      size_t rows = boost::apply_visitor(
          outputParameterVisitor, network[i]).n_rows;

      // Extract from gy the parameters for the i-th network.
      delta = gy.rows(rowCount / channels, (rowCount + rows) / channels - 1);
      delta.reshape(delta.n_rows * channels, delta.n_cols / channels);

      boost::apply_visitor(BackwardVisitor(std::move(
          boost::apply_visitor(outputParameterVisitor,
          network[i])), std::move(delta), std::move(
          boost::apply_visitor(deltaVisitor, network[i]))), network[i]);
      rowCount += rows;
    }

    g = boost::apply_visitor(deltaVisitor, network[0]);
    for (size_t i = 1; i < network.size(); ++i)
    {
      g += boost::apply_visitor(deltaVisitor, network[i]);
    }
    gy.reshape(gy.n_rows * channels, gy.n_cols / channels);
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
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g,
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
  gy.reshape(gy.n_rows / channels, gy.n_cols * channels);

  arma::Mat<eT> delta = gy.rows(rowCount / channels, (rowCount + rows) /
      channels - 1);
  delta.reshape(delta.n_rows * channels, delta.n_cols / channels);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, network[index])), std::move(delta), std::move(
      boost::apply_visitor(deltaVisitor, network[index]))), network[index]);

  // Reshape gy to its original shape.
  gy.reshape(gy.n_rows * channels, gy.n_cols / channels);

  g = boost::apply_visitor(deltaVisitor, network[index]);
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Concat<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& /* gradient */)
{
  if (run)
  {
    size_t rowCount = 0;


    // Reshape error to extract the i-th layer error.
    error.reshape(error.n_rows / channels, error.n_cols * channels);
    for (size_t i = 0; i < network.size(); ++i)
    {
      size_t rows = boost::apply_visitor(
          outputParameterVisitor, network[i]).n_rows;

      // Extract from error the parameters for the i-th network.
      arma::Mat<eT> err = error.rows(rowCount / channels, (rowCount + rows) /
          channels - 1);
      err.reshape(err.n_rows * channels, err.n_cols / channels);

      boost::apply_visitor(GradientVisitor(std::move(input),
          std::move(err)), network[i]);
      rowCount += rows;
    }

    // Reshape error to its original shape.
    error.reshape(error.n_rows * channels, error.n_cols / channels);
  }
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void Concat<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& /* gradient */,
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

  error.reshape(error.n_rows / channels, error.n_cols * channels);
  arma::Mat<eT> err = error.rows(rowCount / channels, (rowCount + rows) /
      channels - 1);
  err.reshape(err.n_rows * channels, err.n_cols / channels);

  boost::apply_visitor(GradientVisitor(std::move(input),
      std::move(err)), network[index]);

  error.reshape(error.n_rows * channels, error.n_cols / channels);
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
