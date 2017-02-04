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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Concat<InputDataType, OutputDataType>::Concat(
    const bool model, const bool same) : model(model), same(same)
{
  parameters.set_size(0, 0);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Concat<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  size_t outSize = 0;

  for (size_t i = 0; i < network.size(); ++i)
  {
    boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
        boost::apply_visitor(outputParameterVisitor, network[i]))),
        network[i]);

    if (boost::apply_visitor(
        outputParameterVisitor, network[i]).n_elem > outSize)
    {
      outSize = boost::apply_visitor(outputParameterVisitor,
          network[i]).n_elem;
    }
  }

  output = arma::zeros(outSize, network.size());
  for (size_t i = 0; i < network.size(); ++i)
  {
    size_t elements = boost::apply_visitor(outputParameterVisitor,
        network[i]).n_elem;

    if (elements < outSize)
    {
      output.submat(0, i, elements - 1, i) = arma::vectorise(
          boost::apply_visitor(outputParameterVisitor, network[i]));
    }
    else
    {
      output.col(i) = arma::vectorise(boost::apply_visitor(
        outputParameterVisitor, network[i]));
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Concat<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  size_t outSize = 0;
  size_t elements = 0;

  for (size_t i = 0, j = 0; i < network.size(); ++i, j += elements)
  {
    elements = boost::apply_visitor(outputParameterVisitor,
        network[i]).n_elem;

    arma::mat delta;
    if (gy.n_cols == 1)
    {
      delta = gy.submat(j, 0, j + elements - 1, 0);
    }
    else
    {
      delta = gy.submat(0, i, elements - 1, i);
    }

    boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, network[i])), std::move(delta), std::move(
        boost::apply_visitor(deltaVisitor, network[i]))), network[i]);

    if (boost::apply_visitor(deltaVisitor, network[i]).n_elem > outSize)
    {
      outSize = boost::apply_visitor(deltaVisitor, network[i]).n_elem;
    }

    if (same)
    {
      if (i == 0)
      {
        g = std::move(boost::apply_visitor(deltaVisitor, network[i]));
      }
      else
      {
        g += std::move(boost::apply_visitor(deltaVisitor, network[i]));
      }
    }
  }

  if (!same)
  {
    g = arma::zeros(outSize, network.size());
    for (size_t i = 0; i < network.size(); ++i)
    {
      size_t elements = boost::apply_visitor(deltaVisitor, network[i]).n_elem;
      if (elements < outSize)
      {
        g.submat(0, i, elements - 1, i) = arma::vectorise(
            boost::apply_visitor(deltaVisitor, network[i]));
      }
      else
      {
        g.col(i) = arma::vectorise(
            boost::apply_visitor(deltaVisitor, network[i]));
      }
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Concat<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& /* input */,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& /* gradient */)
{
  for (size_t i = 0; i < network.size(); ++i)
  {
    boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, network[i])), std::move(error)), network[i]);
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Concat<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */, const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack


#endif
