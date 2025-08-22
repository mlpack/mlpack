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
   * Take an input and convert each patch into rows. Expects that im2row
   * already has the correct shape.
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
    const size_t outputRows = (input.n_rows - dFilterRows + dH) / dH;
    const size_t outputCols = (input.n_cols - dFilterCols + dW) / dW;

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
  template<typename InMatType, typename FilMatType,
      typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, ValidConvolution>, void>
  PadInput(const InMatType& input,
           const FilMatType& /* filter */,
           InMatType& inputPadded,
           const size_t /* dilationW */,
           const size_t /* dilationH */,
           const typename std::enable_if_t<IsMatrix<InMatType>::value>* = 0)
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
  template<typename InMatType, typename FilMatType,
      typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, FullConvolution>, void>
  PadInput(const InMatType& input,
           const FilMatType& filter,
           InMatType& inputPadded,
           const size_t dilationW,
           const size_t dilationH,
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
    inputPadded = InMatType(input.n_rows + 2 * paddingRows,
        input.n_cols + 2 * paddingCols);
    inputPadded.submat(paddingRows, paddingCols, paddingRows + input.n_rows - 1,
        paddingCols + input.n_cols - 1) = input;
  }

  /**
   * Apply padding to an input cube (valid mode).
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param inputPadded Input with padding applied.
   */
  template<typename InCubeType, typename FilMatType,
      typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, ValidConvolution>, void>
  PadInput(const InCubeType& input,
           const FilMatType& /* filter */,
           InCubeType& inputPadded,
           const size_t /* dilationW */,
           const size_t /* dilationH */,
           const typename std::enable_if_t<IsCube<InCubeType>::value>* = 0)
  {
    MakeAlias(inputPadded, input, input.n_rows, input.n_cols, input.n_slices);
  }

  /**
   * Apply padding to an input cube (full mode).
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param inputPadded Input with padding applied.
   */
  template<typename InCubeType, typename FilMatType,
      typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, FullConvolution>, void>
  PadInput(const InCubeType& input,
           const FilMatType& filter,
           InCubeType& inputPadded,
           const size_t dilationW,
           const size_t dilationH,
           const typename std::enable_if_t<IsCube<InCubeType>::value>* = 0)
  {
    // First, compute the necessary padding for the full convolution.  It is
    // possible that this might be an overestimate.  Note that these variables
    // only hold the padding on one side of the input.
    const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
    const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
    const size_t paddingRows = filterRows - 1;
    const size_t paddingCols = filterCols - 1;

    // Pad filter and input to the working output shape.
    inputPadded = InCubeType(input.n_rows + 2 * paddingRows,
        input.n_cols + 2 * paddingCols, input.n_slices);
    inputPadded.subcube(paddingRows, paddingCols, 0,
        paddingRows + input.n_rows - 1, paddingCols + input.n_cols - 1,
        input.n_slices - 1) = input;
  }

  /**
   * Set the size of the output matrix.
   *
   * @param inputPadded Input with padding applied.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   */
  template<typename InMatType, typename FilMatType, typename OutMatType>
  static void
  ComputeOutputSize(const InMatType& inputPadded,
                    const FilMatType& filter,
                    OutMatType& output,
                    const size_t dW = 1,
                    const size_t dH = 1,
                    const size_t dilationW = 1,
                    const size_t dilationH = 1,
                    const typename std::enable_if_t<
                        IsMatrix<OutMatType>::value>* = 0)
  {
    // Compute the output size.  The filterRows and filterCols computation must
    // take into account the fact that dilation only adds rows or columns
    // *between* filter elements.  So, e.g., a dilation of 2 on a kernel size of
    // 3x3 means an effective kernel size of 5x5, *not* 6x6.
    const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
    const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
    const size_t outputRows = (inputPadded.n_rows - filterRows + dH) / dH;
    const size_t outputCols = (inputPadded.n_cols - filterCols + dW) / dW;
    output.zeros(outputRows, outputCols);
  }

  /**
   * Set the size of the output cube. Expects that the output already has the
   * correct amount of slices.
   *
   * @param inputPadded Input with padding applied.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   */
  template<typename InMatType, typename FilMatType, typename OutCubeType>
  static void
  ComputeOutputSize(const InMatType& inputPadded,
                    const FilMatType& filter,
                    OutCubeType& output,
                    const size_t dW = 1,
                    const size_t dH = 1,
                    const size_t dilationW = 1,
                    const size_t dilationH = 1,
                    const typename std::enable_if_t<
                        IsCube<OutCubeType>::value>* = 0)
  {
    // Compute the output size.  The filterRows and filterCols computation must
    // take into account the fact that dilation only adds rows or columns
    // *between* filter elements.  So, e.g., a dilation of 2 on a kernel size of
    // 3x3 means an effective kernel size of 5x5, *not* 6x6.
    const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
    const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
    const size_t outputRows = (inputPadded.n_rows - filterRows + dH) / dH;
    const size_t outputCols = (inputPadded.n_cols - filterCols + dW) / dW;
    output.zeros(outputRows, outputCols, output.n_slices);
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

    if (!appending)
      ComputeOutputSize(inputPadded, filter, output, dW, dH,
          dilationW, dilationH);

    InMatType im2row(output.n_elem, filter.n_elem,
        GetFillType<InMatType>::none);
    Im2Row(inputPadded, im2row, filter.n_rows, filter.n_cols, dW, dH,
        dilationW, dilationH);

    // The filters already have the correct order in memory, just reshape it.
    FilMatType fil2col;
    MakeAlias(fil2col, filter, filter.n_elem, 1);

    // The output is also already in the correct order.
    OutMatType tempOutput;
    MakeAlias(tempOutput, output, output.n_elem, 1);

    tempOutput += im2row * fil2col;
  }

  /**
   * Perform a convolution using 3rd order tensors. Expects that `filter` has
   * `input.n_slices * output.n_slices` slices. The first `input.n_slices`
   * filters output to the first output slice.
   * eg. 2 input slices: filter 0 applies to input 0, output 0,
   * fil 1 -> in 1, out 0, fil 2 -> in 0, out 1, fil 3 -> in 1, out 1,
   * fil 4 -> in 0, out 2, fil 5 -> in 1, out 2, etc.
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

    CubeType inputPadded;
    PadInput(input, filter, inputPadded, dilationW, dilationH);

    if (!appending)
      ComputeOutputSize(inputPadded, filter, output, dW, dH,
          dilationW, dilationH);

    MatType im2row(output.n_rows * output.n_cols, filter.n_rows *
        filter.n_cols * input.n_slices, GetFillType<MatType>::none);
    // Arrange im2row so that each row has patches from each input map.
    #pragma omp parallel for
    for (size_t i = 0; i < input.n_slices; ++i)
    {
      MatType im2rowSv;
      MakeAlias(im2rowSv, im2row, output.n_elem_.n_rows * output.n_cols,
          filter.n_elem_.n_rows * filter.n_cols, output.n_elem_.n_rows *
          output.n_cols * filter.n_elem_.n_rows * filter.n_cols * i);
      Im2Row(inputPadded.slice(i), im2rowSv, filter.n_rows, filter.n_cols,
          dW, dH, dilationW, dilationH);
    }

    const size_t inMaps = input.n_slices;
    const size_t outMaps = filter.n_slices / inMaps;

    // The filters already have the correct order in memory, just reshape it.
    MatType fil2col;
    MakeAlias(fil2col, filter, filter.n_rows * filter.n_cols * inMaps,
        outMaps);

    // The output is also already in the correct order.
    MatType tempOutput;
    MakeAlias(tempOutput, output, output.n_rows * output.n_cols, outMaps);

    tempOutput += im2row * fil2col;
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
    MatType inputPadded;
    PadInput(input, filter, inputPadded, dilationW, dilationH);

    if (!appending)
      ComputeOutputSize(inputPadded, filter, output, dW, dH,
          dilationW, dilationH);

    MatType im2row(output.n_rows * output.n_cols, filter.n_rows *
        filter.n_cols, GetFillType<MatType>::none);
    Im2Row(inputPadded, im2row, filter.n_rows, filter.n_cols, dW, dH,
        dilationW, dilationH);

    // The filters already have the correct order in memory, just reshape it.
    MatType fil2col;
    MakeAlias(fil2col, filter, filter.n_rows * filter.n_cols, filter.n_slices);

    // The output is also already in the correct order.
    MatType tempOutput;
    MakeAlias(tempOutput, output, output.n_rows * output.n_cols,
        filter.n_slices);

    tempOutput += im2row * fil2col;
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
    CubeType inputPadded;
    PadInput(input, filter, inputPadded, dilationW, dilationH);

    if (!appending)
      ComputeOutputSize(inputPadded, filter, output, dW, dH,
          dilationW, dilationH);

    // The filters already have the correct order in memory, just reshape it.
    MatType fil2col;
    MakeAlias(fil2col, filter, filter.n_elem, 1);

    MatType tempOutput;
    MatType im2row(output.n_rows * output.n_cols, filter.n_elem,
        GetFillType<CubeType>::none);

    for (size_t i = 0; i < input.n_slices; ++i)
    {
      MakeAlias(tempOutput, output, output.n_rows * output.n_cols, 1,
          i * output.n_rows * output.n_cols);
      Im2Row(inputPadded.slice(i), im2row, filter.n_rows, filter.n_cols,
          dW, dH, dilationW, dilationH);

      tempOutput += im2row * fil2col;
    }
  }
};  // class NaiveConvolution

} // namespace mlpack

#endif
