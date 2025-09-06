/**
 * @file methods/ann/convolution_rules/fft_convolution.hpp
 * @author Shangtong Zhang
 * @author Marcus Edel
 *
 * Implementation of the convolution through fft.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_CONVOLUTION_RULES_FFT_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_CONVOLUTION_RULES_FFT_CONVOLUTION_HPP

#include <mlpack/prereqs.hpp>
#include "border_modes.hpp"

namespace mlpack {

/**
 * Computes the two-dimensional convolution through fft. This class allows
 * specification of the type of the border type. The convolution can be
 * computed with the valid border type or the full border type (default).
 * Dilation and stride arguments must be equal to one.
 *
 * FullConvolution: returns the full two-dimensional convolution.
 * ValidConvolution: returns only those parts of the convolution that are
 * computed without the zero-padded edges.
 *
 * @tparam BorderMode Type of the border mode (FullConvolution or
 * ValidConvolution).
 * @tparam padLastDim Pad the last dimension of the input to to turn it from
 * odd to even.
 */
template<typename BorderMode = FullConvolution, const bool padLastDim = false>
class FFTConvolution
{
 public:
  /**
   * Perform a convolution through fft. This method only supports input which is
   * even on the last dimension. In case of an odd input width, a user can
   * manually pad the input or specify the padLastDim parameter which takes care
   * of the padding. The filter instead can have any size. When using the valid
   * mode the filter has to be smaller than the input.
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
  template<typename MatType>
  static void Convolution(const MatType& input,
      const MatType& filter,
      MatType& output,
      const size_t dW = 1,
      const size_t dH = 1,
      const size_t dilationW = 1,
      const size_t dilationH = 1,
      const bool appending = false,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    // TODO: Add dilation and stride
    if (dW != 1 || dH != 1)
      throw std::invalid_argument("FFT convolution does not support stride");
    if (dilationW != 1 || dilationH != 1)
      throw std::invalid_argument("FFT convolution does not support dilation");

    MatType inputPadded, filterPadded;
    PadInput(input, filter, inputPadded);
    PadFilter(inputPadded, filter, filterPadded);

    MatType temp = real(ifft2(fft2(inputPadded) % fft2(filterPadded)));

    ExtractOutput(output, temp, input, filter, appending);
  }

  /**
   * Perform a convolution through fft using 3rd order tensors. This method only
   * supports input which is even on the last dimension. In case of an odd input
   * width, a user can manually pad the input or specify the padLastDim
   * parameter which takes care of the padding. The filter instead can have any
   * size. Expects that `filter` has `input.n_slices * output.n_slices` slices.
   * The Nth `input.n_slices` filters are applied to all input slices and output
   * to the Nth output slice.
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
    using ComplexCubeType = typename GetComplexType<CubeType>::type;
    using ComplexMatType = typename GetDenseMatType<ComplexCubeType>::type;

    // TODO: Add dilation and stride
    if (dW != 1 || dH != 1)
      throw std::invalid_argument("FFT convolution does not support stride");
    if (dilationW != 1 || dilationH != 1)
      throw std::invalid_argument("FFT convolution does not support dilation");

    const size_t inMaps = input.n_slices;
    const size_t outMaps = filter.n_slices / inMaps;

    CubeType inputPadded, filterPadded;
    PadInput(input, filter, inputPadded);
    PadFilter(inputPadded, filter, filterPadded);

    // Apply FFT to inputs and filters
    ComplexCubeType fftInput(size(inputPadded));
    ComplexCubeType fftFilter(size(filterPadded));
    for (size_t i = 0; i < inputPadded.n_slices; ++i)
      fftInput.slice(i) = fft2(inputPadded.slice(i));
    for (size_t i = 0; i < filterPadded.n_slices; ++i)
      fftFilter.slice(i) = fft2(filterPadded.slice(i));

    if (!appending)
      ComputeOutputSize(output, input, filter, outMaps);

    // Compute convolution
    for (size_t j = 0; j < output.n_slices; ++j)
    {
      for (size_t i = 0; i < input.n_slices; ++i)
      {
        ComplexMatType& inputSlice = fftInput.slice(i);
        ComplexMatType& filterSlice = fftFilter.slice(j * inMaps + i);
        MatType temp = real(ifft2(inputSlice % filterSlice));

        ExtractOutput(output.slice(j), temp, input, filter, true);
      }
    }
  }

  /**
   * Perform a convolution through fft using dense matrix as input and a 3rd
   * order tensors as filter and output. This method only supports input which
   * is even on the last dimension. In case of an odd input width, a user can
   * manually pad the input or specify the padLastDim parameter which takes care
   * of the padding. The filter instead can have any size.
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
    using ComplexMatType = typename GetComplexType<MatType>::type;

    // TODO: Add dilation and stride
    if (dW != 1 || dH != 1)
      throw std::invalid_argument("FFT convolution does not support stride");
    if (dilationW != 1 || dilationH != 1)
      throw std::invalid_argument("FFT convolution does not support dilation");

    MatType inputPadded;
    CubeType filterPadded;
    PadInput(input, filter, inputPadded);
    PadFilter(inputPadded, filter, filterPadded);

    // Apply FFT to input
    ComplexMatType fftInput = fft2(inputPadded);

    if (!appending)
      ComputeOutputSize(output, input, filter, filter.n_slices);

    // Compute convolution
    for (size_t i = 0; i < filter.n_slices; ++i)
    {
      MatType temp = real(ifft2(fftInput % fft2(filterPadded.slice(i))));

      ExtractOutput(output.slice(i), temp, input, filter, true);
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
   * @param dilationH The dilation factor in y direction.
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
    using ComplexMatType = typename GetComplexType<MatType>::type;

    // TODO: Add dilation and stride
    if (dW != 1 || dH != 1)
      throw std::invalid_argument("FFT convolution does not support stride");
    if (dilationW != 1 || dilationH != 1)
      throw std::invalid_argument("FFT convolution does not support dilation");

    CubeType inputPadded;
    MatType filterPadded;
    PadInput(input, filter, inputPadded);
    PadFilter(inputPadded, filter, filterPadded);

    // Apply FFT to filter
    ComplexMatType fftFilter = fft2(filterPadded);

    if (!appending)
      ComputeOutputSize(output, input, filter, input.n_slices);

    // Compute convolution
    for (size_t i = 0; i < input.n_slices; ++i)
    {
      MatType temp = real(ifft2(fft2(inputPadded.slice(i)) % fftFilter));

      ExtractOutput(output.slice(i), temp, input, filter, true);
    }
  }
 private:
  /**
   * Apply padding to an input matrix.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param inputPadded Input with padding applied.
   */
  template<typename InMatType, typename FilType,
      typename Border = BorderMode>
  static void
  PadInput(const InMatType& input,
           const FilType& filter,
           InMatType& inputPadded,
           const typename std::enable_if_t<IsMatrix<InMatType>::value>* = 0)
  {
    if (std::is_same_v<Border, ValidConvolution>)
    {
      // Use valid padding (none).
      if (!padLastDim)
      {
        MakeAlias(inputPadded, input, input.n_rows, input.n_cols);
      }
      else
      {
        inputPadded = input;
        inputPadded.resize(input.n_rows, input.n_cols + 1);
      }
    }
    else
    {
      // Use full padding.
      const size_t outputRows = input.n_rows + 2 * (filter.n_rows - 1);
      size_t outputCols = input.n_cols + 2 * (filter.n_cols - 1);

      if (padLastDim)
        outputCols++;

      // Pad filter and input to the working output shape.
      inputPadded = zeros<InMatType>(outputRows, outputCols);
      inputPadded.submat(filter.n_rows - 1, filter.n_cols - 1,
          filter.n_rows - 1 + input.n_rows - 1,
          filter.n_cols - 1 + input.n_cols - 1) = input;
    }
  }

  /**
   * Apply padding to an input cube.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param inputPadded Input with padding applied.
   */
  template<typename InCubeType, typename FilType,
      typename Border = BorderMode>
  static void
  PadInput(const InCubeType& input,
           const FilType& filter,
           InCubeType& inputPadded,
           const typename std::enable_if_t<IsCube<InCubeType>::value>* = 0)
  {
    if (std::is_same_v<Border, ValidConvolution>)
    {
      // Use valid padding (none).
      if (!padLastDim)
      {
        MakeAlias(inputPadded, input, input.n_rows, input.n_cols,
            input.n_slices);
      }
      else
      {
        inputPadded = input;
        inputPadded.resize(input.n_rows, input.n_cols + 1, input.n_slices);
      }
    }
    else
    {
      // Use full padding.
      const size_t outputRows = input.n_rows + 2 * (filter.n_rows - 1);
      size_t outputCols = input.n_cols + 2 * (filter.n_cols - 1);

      if (padLastDim)
        outputCols++;

      // Pad filter and input to the working output shape.
      inputPadded = zeros<InCubeType>(outputRows, outputCols, input.n_slices);
      inputPadded.subcube(filter.n_rows - 1, filter.n_cols - 1, 0,
          filter.n_rows - 1 + input.n_rows - 1,
          filter.n_cols - 1 + input.n_cols - 1,
          input.n_slices - 1) = input;
    }
  }

  /**
   * Apply padding to a filter matrix.
   *
   * @param inputPadded Input with padding applied.
   * @param filter Filter used to perform the convolution.
   * @param filterPadded Filter with padding applied.
   */
  template<typename InType, typename FilMatType>
  static void
  PadFilter(const InType& inputPadded,
            const FilMatType& filter,
            FilMatType& filterPadded,
            const typename std::enable_if_t<
                IsMatrix<FilMatType>::value>* = 0)
  {
    filterPadded = filter;
    filterPadded.resize(inputPadded.n_rows, inputPadded.n_cols);
  }

  /**
   * Apply padding to a filter cube.
   *
   * @param inputPadded Input with padding applied.
   * @param filter Filter used to perform the convolution.
   * @param filterPadded Filter with padding applied.
   */
  template<typename InType, typename FilCubeType>
  static void
  PadFilter(const InType& inputPadded,
            const FilCubeType& filter,
            FilCubeType& filterPadded,
            const typename std::enable_if_t<
                IsCube<FilCubeType>::value>* = 0)
  {
    filterPadded = filter;
    filterPadded.resize(inputPadded.n_rows, inputPadded.n_cols,
        filter.n_slices);
  }

  /**
   * Extract the region of interest from the IFFT.
   *
   * @param output Output matrix to extract to.
   * @param temp The output of IFFT.
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param appending If true, it will not initialize the output. Instead,
   *                  it will append the results to the output.
   */
  template<typename OutType, typename InType, typename FilType,
      typename Border = BorderMode>
  static void
  ExtractOutput(OutType& output,
                const OutType& temp,
                const InType& input,
                const FilType& filter,
                bool appending)
  {
    // Extract the region of interest. We don't need to handle the padLastDim in
    // a special way we just cut it out from the output matrix.
    if (std::is_same_v<Border, ValidConvolution>)
    {
      if (appending)
        output += temp.submat(filter.n_rows - 1, filter.n_cols - 1,
            input.n_rows - 1, input.n_cols - 1);
      else
        output = temp.submat(filter.n_rows - 1, filter.n_cols - 1,
            input.n_rows - 1, input.n_cols - 1);
    }
    else
    {
      if (appending)
        output += temp.submat(filter.n_rows - 1, filter.n_cols - 1,
            2 * (filter.n_rows - 1) + input.n_rows - 1,
            2 * (filter.n_cols - 1) + input.n_cols - 1);
      else
        output = temp.submat(filter.n_rows - 1, filter.n_cols - 1,
            2 * (filter.n_rows - 1) + input.n_rows - 1,
            2 * (filter.n_cols - 1) + input.n_cols - 1);
    }
  }

  /**
   * Set the size of the output matrix.
   *
   * @param output Output data that contains the results of the convolution.
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param outSlices The number of slices in the output cube.
   */
  template<typename OutType, typename InType, typename FilType,
      typename Border = BorderMode>
  static void
  ComputeOutputSize(OutType& output,
                    const InType& input,
                    const FilType& filter,
                    const size_t /* outSlices */ = 1,
                    const typename std::enable_if_t<
                        IsMatrix<OutType>::value>* = 0)
  {
    if (std::is_same_v<Border, ValidConvolution>)
    {
      output.zeros(input.n_rows - filter.n_rows + 1,
          input.n_cols - filter.n_cols + 1);
    }
    else
    {
      output.zeros(input.n_rows + filter.n_rows - 1,
          input.n_cols + filter.n_cols - 1);
    }
  }

  /**
   * Set the size of the output cube.
   *
   * @param output Output data that contains the results of the convolution.
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param outSlices The number of slices in the output cube.
   */
  template<typename OutType, typename InType, typename FilType,
      typename Border = BorderMode>
  static void
  ComputeOutputSize(OutType& output,
                    const InType& input,
                    const FilType& filter,
                    const size_t outSlices = 1,
                    const typename std::enable_if_t<
                        IsCube<OutType>::value>* = 0)
  {
    if (std::is_same_v<Border, ValidConvolution>)
    {
      output.zeros(input.n_rows - filter.n_rows + 1,
          input.n_cols - filter.n_cols + 1, outSlices);
    }
    else
    {
      output.zeros(input.n_rows + filter.n_rows - 1,
          input.n_cols + filter.n_cols - 1, outSlices);
    }
  }
};  // class FFTConvolution

} // namespace mlpack

#endif
