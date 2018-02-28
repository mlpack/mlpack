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
 * Computes the two-dimensional convolution. Only ValidConvolution border mode
 * is supported.
 *
 * ValidConvolution: returns the valid two-dimensional convolution.
 * 
 *
 * @tparam BorderMode Type of the border mode (only ValidConvolution
 * supported for now).
 *
 */
template<typename BorderMode = ValidConvolution>
class NativeConvolution
{
 public:
   /*
   * Perform a convolution.
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
            for ( size_t ki = 0; ki < filter.n_rows; ++kernelPtr, ++inputPtr)
              *outputPtr += *kernelPtr * (*inputPtr);
              ki+=dilation+1;
          }
        }
      }
    }

}