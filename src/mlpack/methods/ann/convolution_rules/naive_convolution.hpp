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
class NaiveConvolution : public BaseConvolution<BorderMode>
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
    NaiveConvolution::PadInput(input, filter, inputPadded, dilationW,
        dilationH);

    const size_t inMaps = input.n_slices;
    const size_t outMaps = filter.n_slices / inMaps;

    if (!appending)
      NaiveConvolution::InitalizeOutput(inputPadded, filter, output, dW, dH,
          dilationW, dilationH, outMaps);

    for (size_t i = 0; i < inMaps; i++)
    {
      MatType& inputSlice = inputPadded.slice(i);
      for (size_t j = 0; j < outMaps; j++)
      {
        Conv2(inputSlice, filter.slice(j * inMaps + i), output.slice(j),
            dW, dH, dilationW, dilationH);
      }
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
    MatType inputPadded;
    NaiveConvolution::PadInput(input, filter, inputPadded, dilationW,
        dilationH);

    if (!appending)
      NaiveConvolution::InitalizeOutput(inputPadded, filter, output, dW, dH,
          dilationW, dilationH, filter.n_slices);

    for (size_t s = 0; s < filter.n_slices; s++)
    {
      Conv2(inputPadded, filter.slice(s), output.slice(s), dW, dH,
          dilationW, dilationH);
    }
  }
 private:
  /**
   * Perform a valid convolution.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the convolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilationW The dilation factor in x direction.
   * @param dilationH The dilation factor in y direction.
   */
  template<typename MatType>
  static void Conv2(const MatType& input,
                    const MatType& filter,
                    MatType& output,
                    const size_t dW,
                    const size_t dH,
                    const size_t dilationW,
                    const size_t dilationH)
  {
    using eT = typename MatType::elem_type;

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
};  // class NaiveConvolution

} // namespace mlpack

#endif
