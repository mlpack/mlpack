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
   * Perform a convolution through fft (valid mode). This method only supports
   * input which is even on the last dimension. In case of an odd input width, a
   * user can manually pad the input or specify the padLastDim parameter which
   * takes care of the padding. The filter instead can have any size. When using
   * the valid mode the filter has to be smaller than the input.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename MatType, typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, ValidConvolution>, void>
  Convolution(const MatType& input,
              const MatType& filter,
              MatType& output,
              const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    MatType inputPadded = input;
    MatType filterPadded = filter;

    if (padLastDim)
      inputPadded.resize(inputPadded.n_rows, inputPadded.n_cols + 1);

    // Pad filter and input to the output shape.
    filterPadded.resize(inputPadded.n_rows, inputPadded.n_cols);

    MatType temp = real(ifft2(fft2(inputPadded) % fft2(filterPadded)));

    // Extract the region of interest. We don't need to handle the padLastDim in
    // a special way we just cut it out from the output matrix.
    output = temp.submat(filter.n_rows - 1, filter.n_cols - 1,
        input.n_rows - 1, input.n_cols - 1);
  }

  /**
   * Perform a convolution through fft (full mode). This method only supports
   * input which is even on the last dimension. In case of an odd input width, a
   * user can manually pad the input or specify the padLastDim parameter which
   * takes care of the padding. The filter instead can have any size.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename MatType, typename Border = BorderMode>
  static std::enable_if_t<std::is_same_v<Border, FullConvolution>, void>
  Convolution(const MatType& input,
              const MatType& filter,
              MatType& output,
              const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    // In case of the full convolution outputRows and outputCols doesn't
    // represent the true output size when the padLastDim parameter is set,
    // instead it's the working size.
    const size_t outputRows = input.n_rows + 2 * (filter.n_rows - 1);
    size_t outputCols = input.n_cols + 2 * (filter.n_cols - 1);

    if (padLastDim)
        outputCols++;

    // Pad filter and input to the working output shape.
    MatType inputPadded = zeros<MatType>(outputRows, outputCols);
    inputPadded.submat(filter.n_rows - 1, filter.n_cols - 1,
          filter.n_rows - 1 + input.n_rows - 1,
          filter.n_cols - 1 + input.n_cols - 1) = input;

    MatType filterPadded = filter;
    filterPadded.resize(outputRows, outputCols);

    // Perform FFT and IFFT
    MatType temp = real(ifft2(fft2(inputPadded) % fft2(filterPadded)));

    // Extract the region of interest. We don't need to handle the padLastDim
    // parameter in a special way we just cut it out from the output matrix.
    output = temp.submat(filter.n_rows - 1, filter.n_cols - 1,
        2 * (filter.n_rows - 1) + input.n_rows - 1,
        2 * (filter.n_cols - 1) + input.n_cols - 1);
  }

  /**
   * Perform a convolution through fft using 3rd order tensors. This method only
   * supports input which is even on the last dimension. In case of an odd input
   * width, a user can manually pad the input or specify the padLastDim
   * parameter which takes care of the padding. The filter instead can have any
   * size.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename CubeType>
  static void Convolution(
      const CubeType& input,
      const CubeType& filter,
      CubeType& output,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    using MatType = typename GetDenseMatType<CubeType>::type;
    MatType convOutput;
    FFTConvolution<BorderMode>::Convolution(input.slice(0), filter.slice(0),
        convOutput);

    output = CubeType(convOutput.n_rows, convOutput.n_cols, input.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; ++i)
    {
      FFTConvolution<BorderMode>::Convolution(input.slice(i), filter.slice(i),
          output.slice(i));
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
   */
  template<typename MatType, typename CubeType>
  static void Convolution(const MatType& input,
                          const CubeType& filter,
                          CubeType& output)
  {
    MatType convOutput;
    FFTConvolution<BorderMode>::Convolution(input, filter.slice(0),
        convOutput);

    output = CubeType(convOutput.n_rows, convOutput.n_cols, filter.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < filter.n_slices; ++i)
    {
      FFTConvolution<BorderMode>::Convolution(input, filter.slice(i),
          output.slice(i));
    }
  }

  /**
   * Perform a convolution using a 3rd order tensors as input and output and a
   * dense matrix as filter.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename MatType, typename CubeType>
  static void Convolution(const CubeType& input,
                          const MatType& filter,
                          CubeType& output)
  {
    MatType convOutput;
    FFTConvolution<BorderMode>::Convolution(input.slice(0), filter,
        convOutput);

    output = CubeType(convOutput.n_rows, convOutput.n_cols,
        input.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; ++i)
    {
      FFTConvolution<BorderMode>::Convolution(input.slice(i), filter,
          output.slice(i));
    }
  }
};  // class FFTConvolution

} // namespace mlpack

#endif
