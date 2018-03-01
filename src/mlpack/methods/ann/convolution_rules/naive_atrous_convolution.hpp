/**
 * @file naive_atrous_convolution.hpp
 * @author Aarush Gupta
 *
 * Implementation of atrous convolution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_CONVOLUTION_RULES_NAIVE_ATROUS_CONVOLUTION_HPP
#define MLPACK_METHODS_ANN_CONVOLUTION_RULES_NAIVE_ATROUS_CONVOLUTION_HPP

#include <mlpack/prereqs.hpp>
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
class NaiveAtrousConvolution
{
 public:
   /*
   * Perform a convolution(valid mode).
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilation the dilation factor in both x and y directions.
   */
  template<typename eT, typename Border = BorderMode>
  static typename std::enable_if<
      std::is_same<Border, ValidConvolution>::value, void>::type
  Convolution(const arma::Mat<eT>& input,
              const arma::Mat<eT>& filter,
              arma::Mat<eT>& output,
              const size_t dW = 1,
              const size_t dH = 1,
              const size_t dilation = 0)
    {
      output = arma::zeros<arma::Mat<eT> >((input.n_rows - ((filter.n_rows-1)*dilation + filter.n_rows) + 1) /
      	 dW, (input.n_cols - ((filter.n_cols-1)*dilation + filter.n_cols)+1) / dH);



      eT* outputPtr = output.memptr();

      for (size_t j = 0; j < output.n_cols; ++j)
      {
        for  (size_t i = 0; i < output.n_rows; ++i, outputPtr++)
        {
          const eT* kernelPtr = filter.memptr();
          for (size_t kj = 0; kj < filter.n_cols; kj++)
          {
            const eT* inputPtr = input.colptr(kj + j * dW + kj*(dilation+1)) + i * dH;
            for ( size_t ki = 0; ki < filter.n_rows; ki+=(dilation+1), ++kernelPtr, ++inputPtr)
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
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilation the dilation factor in both x and y directions.

   */
  template<typename eT, typename Border = BorderMode>
  static typename std::enable_if<
      std::is_same<Border, FullConvolution>::value, void>::type
  Convolution(const arma::Mat<eT>& input,
              const arma::Mat<eT>& filter,
              arma::Mat<eT>& output,
              const size_t dW = 1,
              const size_t dH = 1,
              const size_t dilation = 0)
  {
    const size_t outputRows = (input.n_rows + 2 * ((filter.n_rows - 1)*dilation + filter.n_rows - 1)) * dW;
    const size_t outputCols = (input.n_cols + 2 * ((filter.n_cols - 1)*dilation + filter.n_cols - 1)) * dH;

    // Pad filter and input to the working output shape.
    arma::Mat<eT> inputPadded = arma::zeros<arma::Mat<eT> >(outputRows,
        outputCols);
    inputPadded.submat((filter.n_rows - 1)*dilation + filter.n_rows - 1,
        (filter.n_cols - 1)*dilation + filter.n_cols - 1,
        (filter.n_rows - 1)*dilation + filter.n_rows - 1 + input.n_rows - 1,
        (filter.n_cols - 1)*dilation + filter.n_cols - 1 + input.n_cols - 1) = input;

    NaiveAtrousConvolution<ValidConvolution>::Convolution(inputPadded, filter,
        output, 1, 1, dilation);
  }

  /*
   * Perform a convolution using 3rd order tensors.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilation the dilation factor in both x and y directions.
   */
  template<typename eT>
  static void Convolution(const arma::Cube<eT>& input,
                          const arma::Cube<eT>& filter,
                          arma::Cube<eT>& output,
                          const size_t dW = 1,
                          const size_t dH = 1,
                          const size_t dilation = 0)
  {
    arma::Mat<eT> convOutput;
    NaiveAtrousConvolution<BorderMode>::Convolution(input.slice(0), filter.slice(0),
        convOutput, dW, dH, dilation);

    output = arma::Cube<eT>(convOutput.n_rows, convOutput.n_cols,
        input.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; i++)
    {
      NaiveAtrousConvolution<BorderMode>::Convolution(input.slice(i), filter.slice(i),
          output.slice(i), dW, dH, dilation);
    }
  }

  /*
   * Perform a convolution using dense matrix as input and a 3rd order tensors
   * as filter and output.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilation the dilation factor in both x and y directions.
   */
  template<typename eT>
  static void Convolution(const arma::Mat<eT>& input,
                          const arma::Cube<eT>& filter,
                          arma::Cube<eT>& output,
                          const size_t dW = 1,
                          const size_t dH = 1,
                          const size_t dilation = 0)
  {
    arma::Mat<eT> convOutput;
    NaiveAtrousConvolution<BorderMode>::Convolution(input, filter.slice(0),
        convOutput, dW, dH, dilation);

    output = arma::Cube<eT>(convOutput.n_rows, convOutput.n_cols,
        filter.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < filter.n_slices; i++)
    {
      NaiveAtrousConvolution<BorderMode>::Convolution(input, filter.slice(i),
          output.slice(i), dW, dH, dilation);
    }
  }

  /*
   * Perform a convolution using a 3rd order tensors as input and output and a
   * dense matrix as filter.
   *
   * @param input Input used to perform the convolution.
   * @param filter Filter used to perform the conolution.
   * @param output Output data that contains the results of the convolution.
   * @param dW Stride of filter application in the x direction.
   * @param dH Stride of filter application in the y direction.
   * @param dilation the dilation factor in both x and y directions.
   */
  template<typename eT>
  static void Convolution(const arma::Cube<eT>& input,
                          const arma::Mat<eT>& filter,
                          arma::Cube<eT>& output,
                          const size_t dW = 1,
                          const size_t dH = 1,
                          const size_t dilation = 0)
  {
    arma::Mat<eT> convOutput;
    NaiveAtrousConvolution<BorderMode>::Convolution(input.slice(0), filter,
        convOutput, dW, dH, dilation);

    output = arma::Cube<eT>(convOutput.n_rows, convOutput.n_cols,
        input.n_slices);
    output.slice(0) = convOutput;

    for (size_t i = 1; i < input.n_slices; i++)
    {
      NaiveAtrousConvolution<BorderMode>::Convolution(input.slice(i), filter,
          output.slice(i), dW, dH, dilation);
    }
  }
};  // class NaiveAtrousConvolution

} // namespace ann
} // namespace mlpack

#endif
