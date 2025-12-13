/**
 * @file methods/ann/convolution_rules/im2col_convolution.hpp
 * @author Zachary Ng
 *
 * Implementation of the im2col convolution. This is actually im2row because we
 * use column major order.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_CONVOLUTION_RULES_IM2COL_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_CONVOLUTION_RULES_IM2COL_CONVOLUTION_HPP

#include "base_convolution.hpp"

namespace mlpack {

/**
 * Computes the two-dimensional convolution. This class allows specification of
 * the type of the border type. The convolution can be computed with the valid
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
class Im2ColConvolution : public BaseConvolution<BorderMode>
{
 public:
  /**
   * Perform a convolution using 3rd order tensors. Expects that `filter` has
   * `input.n_slices * output.n_slices` slices. The Nth `input.n_slices` filters
   * are applied to all input slices and output to the Nth output slice.
   * eg. 2 input slices: filter 0 applies to input 0, output 0,
   * fil 1 * in 1 = out 0, fil 2 * in 0 = out 1, fil 3 * in 1 = out 1,
   * fil 4 * in 0 = out 2, fil 5 * in 1 = out 2, etc.
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
    Im2ColConvolution::PadInput(input, filter, inputPadded, dilationW,
        dilationH);

    const size_t inMaps = input.n_slices;
    const size_t outMaps = filter.n_slices / inMaps;

    if (!appending)
      Im2ColConvolution::InitalizeOutput(inputPadded, filter, output, dW, dH,
          dilationW, dilationH, outMaps);

    MatType im2row(output.n_rows * output.n_cols, filter.n_rows *
        filter.n_cols * input.n_slices, GetFillType<MatType>::none);
    // Arrange im2row so that each row has patches from each input map.
    for (size_t i = 0; i < input.n_slices; ++i)
    {
      MatType im2rowSv;
      MakeAlias(im2rowSv, im2row, output.n_rows * output.n_cols,
          filter.n_rows * filter.n_cols, output.n_rows *
          output.n_cols * filter.n_rows * filter.n_cols * i);
      Im2Row(inputPadded.slice(i), im2rowSv, filter.n_rows, filter.n_cols,
          dW, dH, dilationW, dilationH);
    }

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
    Im2ColConvolution::PadInput(input, filter, inputPadded, dilationW,
        dilationH);

    if (!appending)
      Im2ColConvolution::InitalizeOutput(inputPadded, filter, output, dW, dH,
          dilationW, dilationH, filter.n_slices);

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
};  // class Im2ColConvolution

} // namespace mlpack

#endif
