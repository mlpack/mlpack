/**
 * @file separable_convolution_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of the Separable Convolution module class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SEPARABLE_CONVOLUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SEPARABLE_CONVOLUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "separable_convolution.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
SeparableConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::SeparableConvolution()
{
  // Nothing to do here.
}

template <
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
SeparableConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::SeparableConvolution(
    const size_t inSize,
    const size_t outSize,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const size_t padW,
    const size_t padH,
    const size_t inputWidth,
    const size_t inputHeight,
    const int numGroups,
    const bool pointWise,
    const std::string &paddingType) :
    inSize(inSize),
    outSize(outSize),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    padWLeft(padW),
    padWRight(padW),
    padHBottom(padH),
    padHTop(padH),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    outputWidth(0),
    outputHeight(0),
    numGroups(numGroups),
    pointWise(pointWise)
{
  if(inSize % numGroups != 0 || outSize % numGroups != 0)
  {
    Log::Fatal << "The output maps / input maps is not possible given "
        << "the number of groups. Input maps / output maps must be " 
        << " divisible by number of groups." << std::endl;
  }
  depthWiseWeights.set_size((inSize * inSize * kernelWidth * kernelHeight) / numGroups +
    inSize, 1);

  pointWiseWeights.set_size((outSize * inSize * kernelWidth * kernelHeight) + outSize,
      1);
  // Transform paddingType to lowercase.
  std::string paddingTypeLow = paddingType;
  std::transform(paddingType.begin(), paddingType.end(), paddingTypeLow.begin(),
      [](unsigned char c){ return std::tolower(c); });

  if (paddingTypeLow == "valid")
  {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingTypeLow == "same")
  {
    InitializeSamePadding();
  }

  padding = ann::Padding<>(padWLeft, padWRight, padHTop, padHBottom); 
}

template <
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
SeparableConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::SeparableConvolution(
    const size_t inSize,
    const size_t outSize,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const std::tuple<size_t, size_t> padW,
    const std::tuple<size_t, size_t> padH,
    const size_t inputWidth,
    const size_t inputHeight,
    const int numGroups,
    const bool pointWise,
    const std::string &paddingType) :
    inSize(inSize),
    outSize(outSize),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    padWLeft(std::get<0>(padW)),
    padWRight(std::get<1>(padW)),
    padHBottom(std::get<1>(padH)),
    padHTop(std::get<0>(padH)),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    outputWidth(0),
    outputHeight(0),
    numGroups(numGroups),
    pointWise(pointWise)
{
  if(inSize % numGroups != 0 || outSize % numGroups != 0)
  {
    Log::Fatal << "The output maps / input maps is not possible given "
        << "the number of groups. Input maps / output maps must be " 
        << " divisible by number of groups." << std::endl;
  }
  depthWiseWeights.set_size((inSize * inSize * kernelWidth * kernelHeight) / numGroups +
    inSize, 1);

  pointWiseWeights.set_size((outSize * inSize * kernelWidth * kernelHeight) + outSize,
      1);
  // Transform paddingType to lowercase.
  std::string paddingTypeLow = paddingType;
  std::transform(paddingType.begin(), paddingType.end(), paddingTypeLow.begin(),
      [](unsigned char c){ return std::tolower(c); });

  if (paddingTypeLow == "valid")
  {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingTypeLow == "same")
  {
    InitializeSamePadding();
  }

  padding = ann::Padding<>(padWLeft, padWRight, padHTop, padHBottom); 
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
void SeparableConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Reset()
{
  depthWiseWeight.set_size(depthWiseWeights.memptr(), kernelWidth, kernelHeight,
        inSize * inSize, false, false);
  pointWiseWeight.set_size(pointWiseWeights.memptr(), 1, 1,
        outSize * inSize, false, false);

  depthWiseBias = arma::mat(depthWiseWeights.memptr() + depthWiseWeight.n_elem,
        inSize, 1, false, false);
  pointWiseBias = arma::mat(pointWiseWeights.memptr() + pointWiseWeight.n_elem,
        outSize, 1, false, false);
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
void SeparableConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  batchSize = input.n_cols;
  inputTemp = arma::cube(const_cast<arma::Mat<eT>&&>(input).memptr(),
      inputWidth, inputHeight, inSize * batchSize, false, false);

  if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
  {
    inputPaddedTemp.set_size(inputTemp.n_rows + padWLeft + padWRight,
        inputTemp.n_cols + padHTop + padHBottom, inputTemp.n_slices);

    for (size_t i = 0; i < inputTemp.n_slices; ++i)
    {
      padding.Forward(std::move(inputTemp.slice(i)),
          std::move(inputPaddedTemp.slice(i)));
    }
  }


}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
void SeparableConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::InitializeSamePadding()
{
  /*
   * Using O = (W - F + 2P) / s + 1;
   */
    size_t totalVerticalPadding = (strideWidth - 1) * inputWidth + kernelWidth -
        strideWidth;
    size_t totalHorizontalPadding = (strideHeight - 1) * inputHeight +
        kernelHeight - strideHeight;

    padWLeft = totalVerticalPadding / 2;
    padWRight = totalVerticalPadding - totalVerticalPadding / 2;
    padHTop = totalHorizontalPadding / 2;
    padHBottom = totalHorizontalPadding - totalHorizontalPadding / 2;
}


} // namespace ann
} // namespace mlpack

#endif