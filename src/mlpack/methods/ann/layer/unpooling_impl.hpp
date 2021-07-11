/**
 * @file methods/ann/layer/unpooling_impl.hpp
 * @author Anjishnu Mukherjee
 *
 * Implementation of the UnPooling class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_UNPOOLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_UNPOOLING_IMPL_HPP

// In case it hasn't yet been included.
#include "unpooling.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
UnPooling<InputDataType, OutputDataType>::UnPooling(
    std::vector<arma::cube> poolingIndices,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight) :
    poolingIndices(poolingIndices),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    inSize(0),
    outSize(0),
    inputWidth(0),
    inputHeight(0),
    outputWidth(0),
    outputHeight(0),
    batchSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void UnPooling<InputDataType, OutputDataType>::Forward(
  const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  batchSize = input.n_cols;

  // inputWidth and inputHeight will be set by respective accessor methods.
  inSize = input.n_elem / (inputWidth * inputHeight * batchSize);
  inputTemp = arma::cube(const_cast<arma::Mat<eT>&>(input).memptr(),
      inputWidth, inputHeight, batchSize * inSize, false, false);

  outputWidth = (inputWidth - 1) * strideWidth + kernelWidth;
  outputHeight = (inputHeight - 1) * strideHeight + kernelHeight;
  outSize = batchSize * inSize;
  outputTemp = arma::zeros<arma::Cube<eT> >(outputWidth, outputHeight,
      outSize);

  for (size_t s = 0; s < inputTemp.n_slices; s++)
  {
    Unpooling(inputTemp.slice(s), outputTemp.slice(s),
        poolingIndices.back().slice(s));
  }

  output = arma::Mat<eT>(outputTemp.memptr(), outputTemp.n_elem / batchSize,
      batchSize);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void UnPooling<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  arma::cube mappedError = arma::cube(((arma::Mat<eT>&) gy).memptr(),
      outputWidth, outputHeight, outSize, false, false);
  gTemp = arma::zeros<arma::cube>(inputWidth, inputHeight, outSize);

  // The contents of this for loop can be re-factored into a private function.
  for (size_t s = 0; s < gTemp.n_slices; s++)
  {
    arma::mat& gySlice = mappedError.slice(s);
    arma::mat& gSlice = gTemp.slice(s);
    arma::mat& idxs = poolingIndices.back().slice(s);

    for (size_t i = 0, j = 0; i < idxs.n_elem; ++i, ++j)
    {
      gSlice(j) = gySlice(idxs(i));
    }
  }
  poolingIndices.pop_back();
  g = arma::mat(gTemp.memptr(), gTemp.n_elem / batchSize, batchSize);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void UnPooling<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(kernelWidth);
  ar & BOOST_SERIALIZATION_NVP(kernelHeight);
  ar & BOOST_SERIALIZATION_NVP(strideWidth);
  ar & BOOST_SERIALIZATION_NVP(strideHeight);
  ar & BOOST_SERIALIZATION_NVP(batchSize);
  ar & BOOST_SERIALIZATION_NVP(inputWidth);
  ar & BOOST_SERIALIZATION_NVP(inputHeight);
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);
}

} // namespace ann
} // namespace mlpack

#endif
