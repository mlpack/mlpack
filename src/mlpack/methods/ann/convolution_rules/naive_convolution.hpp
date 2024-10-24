/**
 * @file methods/ann/convolution_rules/naive_convolution.hpp
 * @author Shangtong Zhang
 * @author Marcus Edel
 *
 * Implementation of the convolution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_CONVOLUTION_RULES_NAIVE_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_CONVOLUTION_RULES_NAIVE_CONVOLUTION_HPP

#include <mlpack/prereqs.hpp>
#include "border_modes.hpp"

namespace mlpack {

/**
 * Computes the two-dimensional convolution. This class allows specification of
 * the type of the border type. The convolution can be compute with the valid
 * border type of the full border type (default).
 *
 * FullConvolution: returns the full two-dimensional convolution.
 * ValidConvolution: returns only those parts of the convolution that are
 * computed without the zero-padded edges.
 *
 * @tparam BorderMode Type of the border mode (FullConvolution or
 * ValidConvolution).
 */
template<typename BorderMode = FullConvolution>
class NaiveConvolution
{
 public:
  /**
   * Perform a convolution (valid mode).
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   * @param appending If true, it will not initialize the output. Instead,
   *                  it will append the results to the output.
   */
  template<typename InMatType, typename FilMatType, typename OutMatType,
      typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, ValidConvolution>, void>
  Convolution(const InMatType& input,
              const FilMatType& filter,
              OutMatType& output,
              const size_t dW = 1,
              const size_t dH = 1,
              const size_t dilationW = 1,
              const size_t dilationH = 1,
              const bool appending = false,
              const typename std::enable_if_t<IsMatrix<InMatType>::value>* = 0)
  {
    using eT = typename InMatType::elem_type;
    // Compute the output size.  The filterRows and filterCols computation must
    // take into account the fact that dilation only adds rows or columns
    // *between* filter elements.  So, e.g., a dilation of 2 on a kernel size of
    // 3x3 means an effective kernel size of 5x5, *not* 6x6.
    if (!appending)
    {
      const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
      const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
      const size_t outputRows = (input.n_rows - filterRows + dH) / dH;
      const size_t outputCols = (input.n_cols - filterCols + dW) / dW;
      output.zeros(outputRows, outputCols);
    }

    // It seems to be about 3.5 times faster to use pointers instead of
    // filter(ki, kj) * input(leftInput + ki, topInput + kj) and output(i, j).
    eT* outputPtr = output.memptr();

    for (size_t j = 0; j < output.n_cols; ++j)
    {
      for (size_t i = 0; i < output.n_rows; ++i, outputPtr++)
      {
        const eT* kernelPtr = filter.memptr();
        for (size_t kj = 0; kj < filter.n_cols; ++kj)
        {
          const eT* inputPtr = input.colptr(kj * dilationW + j * dW) + i * dH;
          for (size_t ki = 0; ki < filter.n_rows; ++ki, ++kernelPtr,
              inputPtr += dilationH)
            *outputPtr += *kernelPtr * (*inputPtr);
        }
      }
    }
  }

  /**
   * Perform a convolution (full mode).
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   * @param appending If true, it will not initialize the output. Instead,
   *                  it will append the results to the output.
   */
  template<typename InMatType, typename FilMatType, typename OutMatType,
      typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, FullConvolution>, void>
  Convolution(const InMatType& input,
              const FilMatType& filter,
              OutMatType& output,
              const size_t dW = 1,
              const size_t dH = 1,
              const size_t dilationW = 1,
              const size_t dilationH = 1,
              const bool appending = false,
              const typename std::enable_if_t<IsMatrix<InMatType>::value>* = 0)
  {
    // First, compute the necessary padding for the full convolution.  It is
    // possible that this might be an overestimate.  Note that these variables
    // only hold the padding on one side of the input.
    const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
    const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
    const size_t paddingRows = filterRows - 1;
    const size_t paddingCols = filterCols - 1;

    // Pad filter and input to the working output shape.
    InMatType inputPadded(input.n_rows + 2 * paddingRows,
        input.n_cols + 2 * paddingCols);
    inputPadded.submat(paddingRows, paddingCols, paddingRows + input.n_rows - 1,
        paddingCols + input.n_cols - 1) = input;

    NaiveConvolution<ValidConvolution>::Convolution(inputPadded, filter,
        output, dW, dH, dilationW, dilationH, appending);
  }

  /**
   * Perform a convolution using 3rd order tensors.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   * @param appending If true, it will not initialize the output. Instead,
   *                  it will append the results to the output.
   */
  template<typename CubeType>
  static void Convolution(
      const CubeType& input,
      const CubeType& filter,
      CubeType& output,
      const size_t dW = 1,
      const size_t dH = 1,
      const size_t dilationW = 1,
      const size_t dilationH = 1,
      const bool appending = false,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    using MatType = typename GetDenseMatType<CubeType>::type;
    MatType convOutput;
    NaiveConvolution<BorderMode>::Convolution(input.slice(0), filter.slice(0),
        convOutput, dW, dH, dilationW, dilationH, appending);

    if (!appending)
      output = CubeType(convOutput.n_rows, convOutput.n_cols, input.n_slices);

    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; ++i)
    {
      NaiveConvolution<BorderMode>::Convolution(input.slice(i), filter.slice(i),
          output.slice(i), dW, dH, dilationW, dilationH, appending);
    }
  }

  /**
   * Perform a convolution using dense matrix as input and a 3rd order tensors
   * as filter and output.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   * @param appending If true, it will not initialize the output. Instead,
   *                  it will append the results to the output.
   */
  template<typename MatType, typename CubeType>
  static void Convolution(
      const MatType& input,
      const CubeType& filter,
      CubeType& output,
      const size_t dW = 1,
      const size_t dH = 1,
      const size_t dilationW = 1,
      const size_t dilationH = 1,
      const bool appending = false,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    MatType convOutput;
    NaiveConvolution<BorderMode>::Convolution(input, filter.slice(0),
        convOutput, dW, dH, dilationW, dilationH, appending);

    if (!appending)
      output = CubeType(convOutput.n_rows, convOutput.n_cols, filter.n_slices);

    output.slice(0) = convOutput;

    for (size_t i = 1; i < filter.n_slices; ++i)
    {
      NaiveConvolution<BorderMode>::Convolution(input, filter.slice(i),
          output.slice(i), dW, dH, dilationW, dilationH, appending);
    }
  }

  /**
   * Perform a convolution using a 3rd order tensors as input and output and a
   * dense matrix as filter.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.x
   * @param appending If true, it will not initialize the output. Instead,
   *                  it will append the results to the output.
   */
  template<typename MatType, typename CubeType>
  static void Convolution(
      const CubeType& input,
      const MatType& filter,
      CubeType& output,
      const size_t dW = 1,
      const size_t dH = 1,
      const size_t dilationW = 1,
      const size_t dilationH = 1,
      const bool appending = false,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    MatType convOutput;
    NaiveConvolution<BorderMode>::Convolution(input.slice(0), filter,
        convOutput, dW, dH, dilationW, dilationH, appending);

    if (!appending)
      output = CubeType(convOutput.n_rows, convOutput.n_cols, input.n_slices);

    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; ++i)
    {
      NaiveConvolution<BorderMode>::Convolution(input.slice(i), filter,
          output.slice(i), dW, dH, dilationW, dilationH, appending);
    }
  }
};  // class NaiveConvolution

} // namespace mlpack

#endif
