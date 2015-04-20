/**
 * @file naive_convolution.hpp
 * @author Shangtong Zhang
 * @author Marcus Edel
 *
 * Implementation of the convolution.
 */
#ifndef __MLPACK_METHODS_ANN_CONVOLUTION_RULES_NAIVE_CONVOLUTION_HPP
#define __MLPACK_METHODS_ANN_CONVOLUTION_RULES_NAIVE_CONVOLUTION_HPP

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
};  // class NaiveConvolution

}; // namespace ann
}; // namespace mlpack

#endif
