/**
 * @file convolution_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Convolution module class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONVOLUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONVOLUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "convolution.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Convolution()
{
  // Nothing to do here.
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Convolution(
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
    const std::string& paddingType) :
    Convolution(
      inSize,
      outSize,
      kernelWidth,
      kernelHeight,
      strideWidth,
      strideHeight,
      std::tuple<size_t, size_t>(padW, padW),
      std::tuple<size_t, size_t>(padH, padH),
      inputWidth,
      inputHeight,
      paddingType)
{
  // Nothing to do here.
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Convolution(
    const size_t inSize,
    const size_t outSize,
    const size_t kernelWidth,
    const size_t kernelHeight,
    const size_t strideWidth,
    const size_t strideHeight,
    const std::tuple<size_t, size_t>& padW,
    const std::tuple<size_t, size_t>& padH,
    const size_t inputWidth,
    const size_t inputHeight,
    const std::string& paddingType) :
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
    outputHeight(0)
{
  weights.set_size((outSize * inSize * kernelWidth * kernelHeight) + outSize,
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
void Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Reset()
{
    weight = arma::cube(weights.memptr(), kernelWidth, kernelHeight,
        outSize * inSize, false, false);
    bias = arma::mat(weights.memptr() + weight.n_elem,
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
void Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  batchSize = input.n_cols;
  arma::cube inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(),
                       inputWidth, inputHeight, inSize * batchSize,
                       false, false);

  if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
  {
    inputPaddedTemp.set_size(inputTemp.n_rows + padWLeft + padWRight,
        inputTemp.n_cols + padHTop + padHBottom, inputTemp.n_slices);

    #pragma omp parallel for
    for (size_t i = 0; i < inputTemp.n_slices; ++i)
    {
      padding.Forward(inputTemp.slice(i), inputPaddedTemp.slice(i));
    }
    inputTemp = inputPaddedTemp;
  }

  size_t wConv = ConvOutSize(inputWidth, kernelWidth, strideWidth, padWLeft,
      padWRight);
  size_t hConv = ConvOutSize(inputHeight, kernelHeight, strideHeight, padHTop,
      padHBottom);
  output.set_size(wConv * hConv * outSize, batchSize);
  outputTemp = arma::Cube<eT>(output.memptr(), wConv, hConv,
                              outSize * batchSize, false, false);
  outputTemp.zeros();
  #pragma omp parallel for
  for (size_t outMap = 0; outMap < outSize * batchSize; outMap++)
  {
    size_t outMapIdx = (outMap % outSize) * inSize, batchCount = outMap/outSize;
    arma::Mat<eT> &curSlice = outputTemp.slice(outMap);

    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      ForwardConvolutionRule::Convolution(inputTemp.slice(inMap +
                                                          (batchCount*inSize)),
                                          weight.slice(outMapIdx),
                                          curSlice, strideWidth, strideHeight,
                                          1, 1, true);
    }
    curSlice += bias(outMap % outSize);
  }

  outputWidth = outputTemp.n_rows;
  outputHeight = outputTemp.n_cols;
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename eT>
void Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Backward(
    const arma::Mat<eT>& input, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  arma::cube mappedError(((arma::Mat<eT>&) gy).memptr(), outputWidth,
      outputHeight, outSize * batchSize, false, false);
  arma::cube inputTemp(((arma::Mat<eT>&) input).memptr(), inputWidth,
      inputHeight, inSize * batchSize, false, false);

  g.set_size(inputTemp.n_rows * inputTemp.n_cols * inSize, batchSize);
  gTemp = arma::Cube<eT>(g.memptr(), inputTemp.n_rows, inputTemp.n_cols,
                         inputTemp.n_slices, false, false);
  gTemp.zeros();
  #pragma omp parallel
  for (size_t outMapIdx = 0; outMapIdx < outSize; outMapIdx++)
  {
    arma::Mat<eT> rotatedFilter;
    Rotate180(weight.slice(outMapIdx), rotatedFilter);
    for (size_t inMap = 0; inMap < inSize; inMap++)
    {
      #pragma omp for
      for (size_t batchCount = 0; batchCount < batchSize; batchCount++) {
        arma::Mat<double> &errSlice = mappedError.slice(outMapIdx +
                                                        batchCount*outSize);

        if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0)
        {
          arma::Mat<eT> output;
          BackwardConvolutionRule::Convolution(errSlice, rotatedFilter, output,
                                               strideWidth, strideHeight);

          arma::subview<eT> sm = output.submat(padWLeft,
                                               padHTop,
                                               padWLeft + gTemp.n_rows - 1,
                                               padHTop + gTemp.n_cols - 1);
          gTemp.slice(inMap + batchCount*inSize) += sm;
        } else {
          BackwardConvolutionRule::Convolution(errSlice, rotatedFilter,
                                               gTemp.slice(inMap +
                                                           batchCount*inSize),
                                               strideWidth, strideHeight,
                                               1, 1, true);
        }
      }
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
template<typename eT>
void Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  arma::cube mappedError(((arma::Mat<eT>&) error).memptr(), outputWidth,
      outputHeight, outSize * batchSize, false, false);
  arma::cube inputTemp;
  if (padWLeft != 0 || padWRight != 0 || padHTop != 0 || padHBottom != 0) {
    inputTemp = inputPaddedTemp;
  } else {
    inputTemp = arma::cube(((arma::Mat<eT>&) input).memptr(), inputWidth,
                           inputHeight, inSize * batchSize, false, false);
  }

  gradient.set_size(weights.n_elem, 1);
  gradientTemp = arma::Cube<eT>(gradient.memptr(), weight.n_rows,
      weight.n_cols, weight.n_slices, false, false);
  gradientTemp.zeros();
  for (size_t outMap = 0; outMap < outSize * batchSize; outMap++)
  {
    size_t outMapIdx = (outMap % outSize) * inSize,
           batchCount = outMap/outSize;
    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      arma::Mat<eT> &inputSlice = inputTemp.slice(inMap+(batchCount*inSize));
      arma::Mat<eT> &deltaSlice = mappedError.slice(outMap);
      arma::Mat<eT> output;
      GradientConvolutionRule::Convolution(inputSlice, deltaSlice,
                                           output, strideWidth, strideHeight);

      if (gradientTemp.n_rows < output.n_rows ||
          gradientTemp.n_cols < output.n_cols)
      {
        gradientTemp.slice(outMapIdx) += output.submat(0, 0,
                                                       gradientTemp.n_rows-1,
                                                       gradientTemp.n_cols-1);
      }
      else if (gradientTemp.n_rows > output.n_rows ||
               gradientTemp.n_cols > output.n_cols)
      {
        gradientTemp.slice(outMapIdx).submat(0, 0,
                                             output.n_rows-1,
                                             output.n_cols-1) += output;
      }
      else
      {
        gradientTemp.slice(outMapIdx) += output;
      }
    }
    gradient.submat(weight.n_elem+(outMap%outSize),
                    0,
                    weight.n_elem+(outMap%outSize),
                    0) = arma::accu(mappedError.slice(outMap));
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
template<typename Archive>
void Convolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::serialize(Archive& ar, const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);
  ar & BOOST_SERIALIZATION_NVP(batchSize);
  ar & BOOST_SERIALIZATION_NVP(kernelWidth);
  ar & BOOST_SERIALIZATION_NVP(kernelHeight);
  ar & BOOST_SERIALIZATION_NVP(strideWidth);
  ar & BOOST_SERIALIZATION_NVP(strideHeight);
  ar & BOOST_SERIALIZATION_NVP(padWLeft);
  ar & BOOST_SERIALIZATION_NVP(padWRight);
  ar & BOOST_SERIALIZATION_NVP(padHBottom);
  ar & BOOST_SERIALIZATION_NVP(padHTop);
  ar & BOOST_SERIALIZATION_NVP(inputWidth);
  ar & BOOST_SERIALIZATION_NVP(inputHeight);
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);

  if (version > 0)
    ar & BOOST_SERIALIZATION_NVP(padding);

  if (Archive::is_loading::value)
  {
    weights.set_size((outSize * inSize * kernelWidth * kernelHeight) + outSize,
        1);
  }
}

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
void Convolution<
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
