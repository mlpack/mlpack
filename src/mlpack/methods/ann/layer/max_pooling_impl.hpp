/**
 * @file max_pooling_impl.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Implementation of the MaxPooling class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MAX_POOLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MAX_POOLING_IMPL_HPP

// In case it hasn't yet been included.
#include "max_pooling.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MaxPooling<InputDataType, OutputDataType>::MaxPooling()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
MaxPooling<InputDataType, OutputDataType>::MaxPooling(
    const size_t kW,
    const size_t kH,
    const size_t dW,
    const size_t dH,
    const bool floor) :
    kW(kW),
    kH(kH),
    dW(dW),
    dH(dH),
    reset(false),
    floor(floor),
    offset(0),
    inputWidth(0),
    inputHeight(0),
    outputWidth(0),
    outputHeight(0),
    deterministic(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MaxPooling<InputDataType, OutputDataType>::Forward(
  const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  const size_t slices = input.n_elem / (inputWidth * inputHeight);
  inputTemp = arma::cube(input.memptr(), inputWidth, inputHeight, slices);

  if (floor)
  {
    outputWidth = std::floor((inputWidth - (double) kW) / (double) dW + 1);
    outputHeight = std::floor((inputHeight - (double) kH) / (double) dH + 1);
    offset = 0;
  }
  else
  {
    outputWidth = std::ceil((inputWidth - (double) kW) / (double) dW + 1);
    outputHeight = std::ceil((inputHeight - (double) kH) / (double) dH + 1);
    offset = 1;
  }

  outputTemp = arma::zeros<arma::Cube<eT> >(outputWidth, outputHeight,
      slices);

  if (!deterministic)
  {
    poolingIndices.push_back(outputTemp);
  }

  if (!reset)
  {
    size_t elements = inputWidth * inputHeight;
    indicesCol = arma::linspace<arma::Col<size_t> >(0, (elements - 1),
        elements);

    indices = arma::Mat<size_t>(indicesCol.memptr(), inputWidth, inputHeight);

    reset = true;
  }

  for (size_t s = 0; s < inputTemp.n_slices; s++)
  {
    if (!deterministic)
    {
      PoolingOperation(inputTemp.slice(s), outputTemp.slice(s),
        poolingIndices.back().slice(s));
    }
    else
    {
      PoolingOperation(inputTemp.slice(s), outputTemp.slice(s),
          inputTemp.slice(s));
    }
  }

  output = arma::Mat<eT>(outputTemp.memptr(), outputTemp.n_elem, 1);

  outputWidth = outputTemp.n_rows;
  outputHeight = outputTemp.n_cols;
  outSize = slices;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MaxPooling<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  arma::cube mappedError = arma::cube(gy.memptr(), outputWidth,
      outputHeight, outSize);

  gTemp = arma::zeros<arma::cube>(inputTemp.n_rows,
      inputTemp.n_cols, inputTemp.n_slices);

  for (size_t s = 0; s < mappedError.n_slices; s++)
  {
    Unpooling(mappedError.slice(s), gTemp.slice(s),
        poolingIndices.back().slice(s));
  }

  poolingIndices.pop_back();

  g = arma::mat(gTemp.memptr(), gTemp.n_elem, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MaxPooling<InputDataType, OutputDataType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & data::CreateNVP(kW, "kW");
  ar & data::CreateNVP(kH, "kH");
  ar & data::CreateNVP(dW, "dW");
  ar & data::CreateNVP(dH, "dH");
}

} // namespace ann
} // namespace mlpack

#endif
