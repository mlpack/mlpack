/**
 * @file methods/ann/convolution_rules/base_convolution.hpp
 * @author Zachary Ng
 *
 * Base class for convolution rules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_CONVOLUTION_RULES_BASE_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_CONVOLUTION_RULES_BASE_CONVOLUTION_HPP

#include <mlpack/prereqs.hpp>
#include "border_modes.hpp"

namespace mlpack {

/**
 * This is an abstract class that contains common functions for convolution.
 * This class allows specification of the type of the border type. The
 * convolution can be computed with the valid border type of the full border
 * type (default).
 *
 * FullConvolution: returns the full two-dimensional convolution.
 * ValidConvolution: returns only those parts of the convolution that are
 * computed without the zero-padded edges.
 *
 * @tparam BorderMode Type of the border mode (FullConvolution or
 * ValidConvolution).
 */
template<typename BorderMode = FullConvolution>
class BaseConvolution
{
 protected:
  /**
   * Apply padding to an input matrix.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param inputPadded Input with padding applied.
   */
  template<typename InMatType, typename FilType, typename Border = BorderMode>
  static void
  PadInput(const InMatType& input,
           const FilType& filter,
           InMatType& inputPadded,
           const size_t dilationW,
           const size_t dilationH,
           const typename std::enable_if_t<IsMatrix<InMatType>::value>* = 0)
  {
    if constexpr (std::is_same_v<Border, ValidConvolution>)
    {
      // Use valid padding (none).
      MakeAlias(inputPadded, input, input.n_rows, input.n_cols);
    }
    else
    {
      // Use full padding

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
      inputPadded.submat(paddingRows, paddingCols,
          paddingRows + input.n_rows - 1,
          paddingCols + input.n_cols - 1) = input;
    }
  }

  /**
   * Apply padding to an input cube.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param inputPadded Input with padding applied.
   */
  template<typename InCubeType, typename FilType, typename Border = BorderMode>
  static void
  PadInput(const InCubeType& input,
           const FilType& filter,
           InCubeType& inputPadded,
           const size_t dilationW,
           const size_t dilationH,
           const typename std::enable_if_t<IsCube<InCubeType>::value>* = 0)
  {
    if constexpr (std::is_same_v<Border, ValidConvolution>)
    {
      // Use valid padding (none).
      MakeAlias(inputPadded, input, input.n_rows, input.n_cols, input.n_slices);
    }
    else
    {
      // Use full padding

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
  }

  /**
   * Initalize the output to the required size.
   *
   * @param inputPadded Input with padding applied.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   * @param outSlices The number of slices in the output cube.
   */
  template<typename InMatType, typename FilType, typename OutMatType>
  static void
  InitalizeOutput(const InMatType& inputPadded,
                  const FilType& filter,
                  OutMatType& output,
                  const size_t dW = 1,
                  const size_t dH = 1,
                  const size_t dilationW = 1,
                  const size_t dilationH = 1,
                  const size_t /* outSlices */ = 1,
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
   * Initalize the output to the required size.
   *
   * @param inputPadded Input with padding applied.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   * @param outSlices The number of slices in the output cube.
   */
  template<typename InMatType, typename FilType, typename OutCubeType>
  static void
  InitalizeOutput(const InMatType& inputPadded,
                  const FilType& filter,
                  OutCubeType& output,
                  const size_t dW = 1,
                  const size_t dH = 1,
                  const size_t dilationW = 1,
                  const size_t dilationH = 1,
                  const size_t outSlices = 1,
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
    output.zeros(outputRows, outputCols, outSlices);
  }
};  // class BaseConvolution

} // namespace mlpack

#endif
