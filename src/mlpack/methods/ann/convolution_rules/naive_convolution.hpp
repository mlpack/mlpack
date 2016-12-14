/**
 * @file naive_convolution.hpp
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

#include <mlpack/core.hpp>
#include "border_modes.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

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
  /*
   * Perform a convolution (valid mode).
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename eT, typename Border = BorderMode>
  static typename std::enable_if<
      std::is_same<Border, ValidConvolution>::value, void>::type
  Convolution(const arma::Mat<eT>& input,
              const arma::Mat<eT>& filter,
              arma::Mat<eT>& output)
  {
    output = arma::zeros<arma::Mat<eT> >(input.n_rows - filter.n_rows + 1,
        input.n_cols - filter.n_cols + 1);

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
          const eT* inputPtr = input.colptr(kj + j) + i;
          for (size_t ki = 0; ki < filter.n_rows; ++ki, ++kernelPtr, ++inputPtr)
            *outputPtr += *kernelPtr * (*inputPtr);
        }
      }
    }
  }

  /*
   * Perform a convolution (full mode).
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename eT, typename Border = BorderMode>
  static typename std::enable_if<
      std::is_same<Border, FullConvolution>::value, void>::type
  Convolution(const arma::Mat<eT>& input,
              const arma::Mat<eT>& filter,
              arma::Mat<eT>& output)
  {
    const size_t outputRows = input.n_rows + 2 * (filter.n_rows - 1);
    const size_t outputCols = input.n_cols + 2 * (filter.n_cols - 1);

    // Pad filter and input to the working output shape.
    arma::Mat<eT> inputPadded = arma::zeros<arma::Mat<eT> >(outputRows,
        outputCols);
    inputPadded.submat(filter.n_rows - 1, filter.n_cols - 1,
          filter.n_rows - 1 + input.n_rows - 1,
          filter.n_cols - 1 + input.n_cols - 1) = input;

    NaiveConvolution<ValidConvolution>::Convolution(inputPadded, filter,
        output);
  }

  /*
   * Perform a convolution using 3rd order tensors.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename eT>
  static void Convolution(const arma::Cube<eT>& input,
                          const arma::Cube<eT>& filter,
                          arma::Cube<eT>& output)
  {
    arma::Mat<eT> convOutput;
    NaiveConvolution<BorderMode>::Convolution(input.slice(0), filter.slice(0),
        convOutput);

    output = arma::Cube<eT>(convOutput.n_rows, convOutput.n_cols,
        input.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; i++)
    {
      NaiveConvolution<BorderMode>::Convolution(input.slice(i), filter.slice(i),
          output.slice(i));
    }
  }

  /*
   * Perform a convolution using dense matrix as input and a 3rd order tensors
   * as filter and output.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename eT>
  static void Convolution(const arma::Mat<eT>& input,
                          const arma::Cube<eT>& filter,
                          arma::Cube<eT>& output)
  {
    arma::Mat<eT> convOutput;
    NaiveConvolution<BorderMode>::Convolution(input, filter.slice(0),
        convOutput);

    output = arma::Cube<eT>(convOutput.n_rows, convOutput.n_cols,
        filter.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < filter.n_slices; i++)
    {
      NaiveConvolution<BorderMode>::Convolution(input, filter.slice(i),
          output.slice(i));
    }
  }

  /*
   * Perform a convolution using a 3rd order tensors as input and output and a
   * dense matrix as filter.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   */
  template<typename eT>
  static void Convolution(const arma::Cube<eT>& input,
                          const arma::Mat<eT>& filter,
                          arma::Cube<eT>& output)
  {
    arma::Mat<eT> convOutput;
    NaiveConvolution<BorderMode>::Convolution(input.slice(0), filter,
        convOutput);

    output = arma::Cube<eT>(convOutput.n_rows, convOutput.n_cols,
        input.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; i++)
    {
      NaiveConvolution<BorderMode>::Convolution(input.slice(i), filter,
          output.slice(i));
    }
  }

};  // class NaiveConvolution

} // namespace ann
} // namespace mlpack

#endif
