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
 private:
  /**
   * Take an input and convert each patch into rows.
   *
   * @param input Input used to perform the convolution.
   * @param im2row Patches of the input as rows.
   * @param filterRows Number of rows in a filter.
   * @param filterCols Number of columns in a filter.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   */
  template<typename MatType>
  static void Im2Row(const MatType& input,
                     MatType& im2row,
                     const size_t filterRows,
                     const size_t filterCols,
                     const size_t dW = 1,
                     const size_t dH = 1,
                     const size_t dilationW = 1,
                     const size_t dilationH = 1)
  {
    const size_t dFilterRows = filterRows * dilationH - (dilationH - 1);
    const size_t dFilterCols = filterCols * dilationW - (dilationW - 1);
    const size_t filterElems = filterRows * filterCols;
    const size_t outputRows = (input.n_rows - dFilterRows + dH) / dH;
    const size_t outputCols = (input.n_cols - dFilterCols + dW) / dW;
    const size_t outputElems = outputRows * outputCols;

    // output = im2row * fil2col.
    // im2row has `outputElems` rows and `filterElems` cols so that output has
    // `outputElems` rows and `outMaps` cols. Each output column is a slice of the
    // output cube.
    im2row = MatType(outputElems, filterElems, GetFillType<MatType>::none);

    // (i, j) = top left corner of a patch
    for (size_t j = 0; j < outputCols; j++)
    {
      for (size_t i = 0; i < outputRows; i++)
      {
        size_t nRow = j * outputRows + i;
        size_t nCol = 0;
        for (size_t kj = 0; kj < filterCols; kj++)
        {
          for (size_t ki = 0; ki < filterRows; ki++)
          {
            im2row.at(nRow, nCol) = input.at(i * dH + ki * dilationH,
                j * dW + kj * dilationW);
            nCol++;
          }
        }
      }
    }
  }

  /**
   * Apply padding to an input matrix (valid mode).
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param inputPadded Input with padding applied.
   */
  template<typename InMatType, typename FilMatType, typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, ValidConvolution>, void>
  PadInput(const InMatType& input,
           const FilMatType& /* filter */,
           InMatType& inputPadded,
           const size_t /* dilationW */,
           const size_t /* dilationH */)
  {
    MakeAlias(inputPadded, input, input.n_rows, input.n_cols);
  }

  /**
   * Apply padding to an input matrix (full mode).
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param inputPadded Input with padding applied.
   */
  template<typename InMatType, typename FilMatType, typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, FullConvolution>, void>
  PadInput(const InMatType& input,
           const FilMatType& filter,
           InMatType& inputPadded,
           const size_t dilationW,
           const size_t dilationH)
  {
    // First, compute the necessary padding for the full convolution.  It is
    // possible that this might be an overestimate.  Note that these variables
    // only hold the padding on one side of the input.
    const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
    const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
    const size_t paddingRows = filterRows - 1;
    const size_t paddingCols = filterCols - 1;

    // Pad filter and input to the working output shape.
    inputPadded = InMatType(input.n_rows + 2 * paddingRows,
        input.n_cols + 2 * paddingCols);
    inputPadded.submat(paddingRows, paddingCols, paddingRows + input.n_rows - 1,
        paddingCols + input.n_cols - 1) = input;
  }

 public:
  /**
   * Perform a convolution with dense matrices.
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
  template<typename InMatType, typename FilMatType, typename OutMatType>
  static void
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
    InMatType inputPadded;
    PadInput(input, filter, inputPadded, dilationW, dilationH);
    // Compute the output size.  The filterRows and filterCols computation must
    // take into account the fact that dilation only adds rows or columns
    // *between* filter elements.  So, e.g., a dilation of 2 on a kernel size of
    // 3x3 means an effective kernel size of 5x5, *not* 6x6.
    if (!appending)
    {
      const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
      const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
      const size_t outputRows = (inputPadded.n_rows - filterRows + dH) / dH;
      const size_t outputCols = (inputPadded.n_cols - filterCols + dW) / dW;
      output.zeros(outputRows, outputCols);
    }

    InMatType im2row;
    Im2Row(inputPadded, im2row, filter.n_rows, filter.n_cols, dW, dH,
        dilationW, dilationH);

    FilMatType fil2col;
    MakeAlias(fil2col, filter, filter.n_elem, 1);

    OutMatType outputTmp;
    MakeAlias(outputTmp, output, output.n_elem, 1);

    outputTmp += im2row * fil2col;
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
