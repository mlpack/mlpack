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
  // Vertically concatentate output from each layer.
  output = boost::apply_visitor(outputParameterVisitor, network.front());
  for (size_t i = 1; i < network.size(); ++i)
  {
    output = arma::join_cols(output,
        boost::apply_visitor(outputParameterVisitor, network[i]));
  }
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
    arma::mat delta;
    for (size_t i = 0; i < network.size(); ++i)
    {
      // Use rows from the error corresponding to the output from each layer.
      size_t rows = boost::apply_visitor(
          outputParameterVisitor, network[i]).n_rows;
      delta = gy.rows(rowCount, rowCount + rows - 1);
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
    rowCount += boost::apply_visitor(outputParameterVisitor, network[i]).n_rows;
  }
  rows = boost::apply_visitor(outputParameterVisitor, network[index]).n_rows;
  arma::mat delta = gy.rows(rowCount, rowCount + rows - 1);
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, network[index])), std::move(delta), std::move(
      boost::apply_visitor(deltaVisitor, network[index]))), network[index]);

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
    for (size_t i = 0; i < network.size(); ++i)
    {
      size_t rows = boost::apply_visitor(
          outputParameterVisitor, network[i]).n_rows;
      boost::apply_visitor(GradientVisitor(std::move(input),
          std::move(error.rows(rowCount, rowCount + rows - 1))), network[i]);
      rowCount += rows;
    }
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
    rowCount += boost::apply_visitor(outputParameterVisitor, network[i]).n_rows;
  }
  size_t rows = boost::apply_visitor(
      outputParameterVisitor, network[index]).n_rows;
  boost::apply_visitor(GradientVisitor(std::move(input),
      std::move(error.rows(rowCount, rowCount + rows - 1))), network[index]);
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
