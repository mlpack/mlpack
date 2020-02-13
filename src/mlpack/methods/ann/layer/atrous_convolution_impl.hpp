/**
 * @file atrous_convolution_impl.hpp
 * @author Aarush Gupta
 * @author Shikhar Jaiswal
 *
 * Implementation of the Atrous Convolution module class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ATROUS_CONVOLUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ATROUS_CONVOLUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "atrous_convolution.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
AtrousConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::AtrousConvolution()
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
AtrousConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::AtrousConvolution(
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
    const size_t dilationWidth,
    const size_t dilationHeight,
    const std::string& paddingType) :
    AtrousConvolution(
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
      dilationWidth,
      dilationHeight,
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
AtrousConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::AtrousConvolution(
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
    const size_t dilationWidth,
    const size_t dilationHeight,
    const std::string& paddingType) :
    inSize(inSize),
    outSize(outSize),
    kernelWidth(kernelWidth),
    kernelHeight(kernelHeight),
    strideWidth(strideWidth),
    strideHeight(strideHeight),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    outputWidth(0),
    outputHeight(0),
    dilationWidth(dilationWidth),
    dilationHeight(dilationHeight)
{
  weights.set_size((outSize * inSize * kernelWidth * kernelHeight) + outSize,
      1);

  // Transform paddingType to lowercase.
  std::string paddingTypeLow = paddingType;
  std::transform(paddingType.begin(), paddingType.end(), paddingTypeLow.begin(),
      [](unsigned char c){ return std::tolower(c); });

  size_t padWLeft = std::get<0>(padW);
  size_t padWRight = std::get<1>(padW);
  size_t padHTop = std::get<0>(padH);
  size_t padHBottom = std::get<1>(padH);
  if (paddingTypeLow == "valid")
  {
    padWLeft = 0;
    padWRight = 0;
    padHTop = 0;
    padHBottom = 0;
  }
  else if (paddingTypeLow == "same")
  {
    InitializeSamePadding(padWLeft, padWRight, padHTop, padHBottom);
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
void AtrousConvolution<
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
void AtrousConvolution<
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

  if (padding.PadWLeft() != 0 || padding.PadWRight() != 0 ||
      padding.PadHTop() != 0 || padding.PadHBottom() != 0)
  {
    inputPaddedTemp.set_size(
        inputTemp.n_rows + padding.PadWLeft() + padding.PadWRight(),
        inputTemp.n_cols + padding.PadHTop() + padding.PadHBottom(),
        inputTemp.n_slices);

    for (size_t i = 0; i < inputTemp.n_slices; ++i)
    {
      padding.Forward(std::move(inputTemp.slice(i)),
          std::move(inputPaddedTemp.slice(i)));
    }
  }

  size_t wConv = ConvOutSize(inputWidth, kernelWidth, strideWidth,
      padding.PadWLeft(), padding.PadWRight(), dilationWidth);
  size_t hConv = ConvOutSize(inputHeight, kernelHeight, strideHeight,
      padding.PadHTop(), padding.PadHBottom(), dilationHeight);

  output.set_size(wConv * hConv * outSize, batchSize);
  outputTemp = arma::Cube<eT>(output.memptr(), wConv, hConv,
      outSize * batchSize, false, false);
  outputTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0, batchCount = 0; outMap <
      outSize * batchSize; outMap++)
  {
    if (outMap != 0 && outMap % outSize == 0)
    {
      batchCount++;
      outMapIdx = 0;
    }

    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      arma::Mat<eT> convOutput;

      if (padding.PadWLeft() != 0 || padding.PadWRight() != 0 ||
          padding.PadHTop() != 0 || padding.PadHBottom() != 0)
      {
        ForwardConvolutionRule::Convolution(inputPaddedTemp.slice(inMap +
            batchCount * inSize), weight.slice(outMapIdx), convOutput,
            strideWidth, strideHeight, dilationWidth, dilationHeight);
      }
      else
      {
        ForwardConvolutionRule::Convolution(inputTemp.slice(inMap +
            batchCount * inSize), weight.slice(outMapIdx), convOutput,
            strideWidth, strideHeight, dilationWidth, dilationHeight);
      }

      outputTemp.slice(outMap) += convOutput;
    }

    outputTemp.slice(outMap) += bias(outMap % outSize);
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
void AtrousConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  arma::cube mappedError(gy.memptr(), outputWidth, outputHeight,
      outSize * batchSize, false, false);

  g.set_size(inputTemp.n_rows * inputTemp.n_cols * inSize, batchSize);
  gTemp = arma::Cube<eT>(g.memptr(), inputTemp.n_rows,
      inputTemp.n_cols, inputTemp.n_slices, false, false);
  gTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0, batchCount = 0; outMap <
      outSize * batchSize; outMap++)
  {
    if (outMap != 0 && outMap % outSize == 0)
    {
      batchCount++;
      outMapIdx = 0;
    }

    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      arma::Mat<eT> output, rotatedFilter;
      Rotate180(weight.slice(outMapIdx), rotatedFilter);

      BackwardConvolutionRule::Convolution(mappedError.slice(outMap),
          rotatedFilter, output, strideWidth, strideHeight, dilationWidth,
          dilationHeight);

      if (padding.PadWLeft() != 0 || padding.PadWRight() != 0 ||
          padding.PadHTop() != 0 || padding.PadHBottom() != 0)
      {
        gTemp.slice(inMap + batchCount * inSize) +=
            output.submat(padding.PadWLeft(), padding.PadHTop(),
                          padding.PadWLeft() + gTemp.n_rows - 1,
                          padding.PadHTop() + gTemp.n_cols - 1);
      }
      else
      {
        gTemp.slice(inMap + batchCount * inSize) += output;
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
void AtrousConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::Gradient(
    const arma::Mat<eT>&& /* input */,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  arma::cube mappedError(error.memptr(), outputWidth, outputHeight,
      outSize * batchSize, false, false);

  gradient.set_size(weights.n_elem, 1);
  gradientTemp = arma::Cube<eT>(gradient.memptr(), weight.n_rows,
      weight.n_cols, weight.n_slices, false, false);
  gradientTemp.zeros();

  for (size_t outMap = 0, outMapIdx = 0, batchCount = 0; outMap <
      outSize * batchSize; outMap++)
  {
    if (outMap != 0 && outMap % outSize == 0)
    {
      batchCount++;
      outMapIdx = 0;
    }

    for (size_t inMap = 0; inMap < inSize; inMap++, outMapIdx++)
    {
      arma::Mat<eT> inputSlice;
      if (padding.PadWLeft() != 0 || padding.PadWRight() != 0 ||
          padding.PadHTop() != 0 || padding.PadHBottom() != 0)
      {
        inputSlice = inputPaddedTemp.slice(inMap + batchCount * inSize);
      }
      else
      {
        inputSlice = inputTemp.slice(inMap + batchCount * inSize);
      }

      arma::Mat<eT> deltaSlice = mappedError.slice(outMap);

      arma::Mat<eT> output;
      GradientConvolutionRule::Convolution(inputSlice, deltaSlice,
          output, strideWidth, strideHeight, 1, 1);

      if (dilationHeight > 1)
      {
        for (size_t i = 1; i < output.n_cols; i++){
          output.shed_cols(i, i + dilationHeight - 2);
        }
      }
      if (dilationWidth > 1)
      {
        for (size_t i = 1; i < output.n_rows; i++){
          output.shed_rows(i, i + dilationWidth - 2);
        }
      }

      if (gradientTemp.n_rows < output.n_rows ||
          gradientTemp.n_cols < output.n_cols)
      {
        gradientTemp.slice(outMapIdx) += output.submat(0, 0,
            gradientTemp.n_rows - 1, gradientTemp.n_cols - 1);
      }
      else if (gradientTemp.n_rows > output.n_rows ||
          gradientTemp.n_cols > output.n_cols)
      {
        gradientTemp.slice(outMapIdx).submat(0, 0, output.n_rows - 1,
            output.n_cols - 1) += output;
      }
      else
      {
        gradientTemp.slice(outMapIdx) += output;
      }
    }

    gradient.submat(weight.n_elem + (outMap % outSize), 0, weight.n_elem +
        (outMap % outSize), 0) = arma::accu(mappedError.slice(outMap));
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
void AtrousConvolution<
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

  // These are now stored in the padding layer.
  if (version < 2 && Archive::is_loading::value)
  {
    size_t padWLeft, padWRight, padHBottom, padHTop;
    ar & BOOST_SERIALIZATION_NVP(padWLeft);
    ar & BOOST_SERIALIZATION_NVP(padWRight);
    ar & BOOST_SERIALIZATION_NVP(padHBottom);
    ar & BOOST_SERIALIZATION_NVP(padHTop);
  }

  ar & BOOST_SERIALIZATION_NVP(inputWidth);
  ar & BOOST_SERIALIZATION_NVP(inputHeight);
  ar & BOOST_SERIALIZATION_NVP(outputWidth);
  ar & BOOST_SERIALIZATION_NVP(outputHeight);
  ar & BOOST_SERIALIZATION_NVP(dilationWidth);
  ar & BOOST_SERIALIZATION_NVP(dilationHeight);

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
void AtrousConvolution<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    InputDataType,
    OutputDataType
>::InitializeSamePadding(size_t& padWLeft,
                         size_t& padWRight,
                         size_t& padHTop,
                         size_t& padHBottom) const
{
  /*
   * Using O = (W - F + 2P) / s + 1;
   */
  size_t totalVerticalPadding = (strideWidth - 1) * inputWidth + kernelWidth -
      strideWidth + (dilationWidth - 1) * (kernelWidth - 1);
  size_t totalHorizontalPadding = (strideHeight - 1) * inputHeight +
      kernelHeight - strideHeight + (dilationHeight - 1) * (kernelHeight - 1);

  padWLeft = totalVerticalPadding / 2;
  padWRight = totalVerticalPadding - totalVerticalPadding / 2;
  padHTop = totalHorizontalPadding / 2;
  padHBottom = totalHorizontalPadding - totalHorizontalPadding / 2;
}

} // namespace ann
} // namespace mlpack

#endif
