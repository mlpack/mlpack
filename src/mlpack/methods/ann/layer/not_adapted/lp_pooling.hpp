/**
 * @file methods/ann/layer/lp_pooling.hpp
 * @author Abhinav Anan
 *
 * Definition of the LpPooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LP_POOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_LP_POOLING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of the LPPooling.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class LpPooling
{
 public:
  //! Create the LpPooling object.
  LpPooling();

  /**
   * Create the LpPooling object using the specified number of units.
   *
   * @param normType Parameter for type of norm.
   * @param kernelWidth Width of the pooling window.
   * @param kernelHeight Height of the pooling window.
   * @param strideWidth Width of the stride operation.
   * @param strideHeight Width of the stride operation.
   * @param floor Set to true to use floor method.
   */
  LpPooling(const size_t normType,
            const size_t kernelWidth,
            const size_t kernelHeight,
            const size_t strideWidth = 1,
            const size_t strideHeight = 1,
            const bool floor = true);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the intput width.
  size_t const& InputWidth() const { return inputWidth; }
  //! Modify the input width.
  size_t& InputWidth() { return inputWidth; }

  //! Get the input height.
  size_t const& InputHeight() const { return inputHeight; }
  //! Modify the input height.
  size_t& InputHeight() { return inputHeight; }

  //! Get the output width.
  size_t const& OutputWidth() const { return outputWidth; }
  //! Modify the output width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the output height.
  size_t const& OutputHeight() const { return outputHeight; }
  //! Modify the output height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the input size.
  size_t InputSize() const { return inSize; }

  //! Get the output size.
  size_t OutputSize() const { return outSize; }

  //! Get the normType.
  size_t NormType() const { return normType; }
  //! Modify the normType.
  size_t& NormType() { return normType; }

  //! Get the kernel width.
  size_t KernelWidth() const { return kernelWidth; }
  //! Modify the kernel width.
  size_t& KernelWidth() { return kernelWidth; }

  //! Get the kernel height.
  size_t KernelHeight() const { return kernelHeight; }
  //! Modify the kernel height.
  size_t& KernelHeight() { return kernelHeight; }

  //! Get the stride width.
  size_t StrideWidth() const { return strideWidth; }
  //! Modify the stride width.
  size_t& StrideWidth() { return strideWidth; }

  //! Get the stride height.
  size_t StrideHeight() const { return strideHeight; }
  //! Modify the stride height.
  size_t& StrideHeight() { return strideHeight; }

  //! Get the value of the rounding operation
  bool const& Floor() const { return floor; }
  //! Modify the value of the rounding operation
  bool& Floor() { return floor; }

  //! Get the size of the weights.
  size_t WeightSize() const { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Apply pooling to the input and store the results.
   *
   * @param input The input to be apply the pooling rule.
   * @param output The pooled result.
   */
  template<typename eT>
  void Pooling(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    arma::Mat<eT> inputPre = input;
    inputPre = pow(inputPre, normType);

    for (size_t i = 1; i < input.n_cols; ++i)
      inputPre.col(i) += inputPre.col(i - 1);

    for (size_t i = 1; i < input.n_rows; ++i)
      inputPre.row(i) += inputPre.row(i - 1);

    for (size_t j = 0, colidx = 0; j < output.n_cols;
         ++j, colidx += strideHeight)
    {
      for (size_t i = 0, rowidx = 0; i < output.n_rows;
           ++i, rowidx += strideWidth)
      {
        double val = 0.0;
        size_t rowEnd = rowidx + kernelWidth - 1;
        size_t colEnd = colidx + kernelHeight - 1;

        if (rowEnd > input.n_rows - 1)
          rowEnd = input.n_rows - 1;
        if (colEnd > input.n_cols - 1)
          colEnd = input.n_cols - 1;

        val += inputPre(rowEnd, colEnd);
        if (rowidx >= 1)
        {
          if (colidx >= 1)
            val += inputPre(rowidx - 1, colidx - 1);
          val -= inputPre(rowidx - 1, colEnd);
        }

        if (colidx >= 1)
          val -= inputPre(rowEnd, colidx - 1);

        output(i, j) = val;
      }
    }

    output = pow(output, 1.0 / normType);
  }

  /**
   * Apply unpooling to the input and store the results.
   *
   * @param input The input to be apply the unpooling rule.
   * @param output The pooled result.
   */
  template<typename eT>
  void Unpooling(const arma::Mat<eT>& input,
                 const arma::Mat<eT>& error,
                 arma::Mat<eT>& output)
  {
    arma::Mat<eT> unpooledError;
    for (size_t j = 0, colidx = 0; j < input.n_cols; j += strideHeight,
         colidx++)
    {
      for (size_t i = 0, rowidx = 0; i < input.n_rows; i += strideWidth,
           rowidx++)
      {
        size_t rowEnd = i + kernelWidth - 1;
        size_t colEnd = j + kernelHeight - 1;

        if (rowEnd > input.n_rows - 1)
        {
          if (floor)
            continue;
          rowEnd = input.n_rows - 1;
        }

        if (colEnd > input.n_cols - 1)
        {
          if (floor)
            continue;
          colEnd = input.n_cols - 1;
        }

        arma::mat InputArea = input(arma::span(i, rowEnd),
            arma::span(j, colEnd));

        size_t sum = pow(accu(pow(InputArea, normType)),
            (normType - 1) / normType);
        unpooledError = arma::Mat<eT>(InputArea.n_rows, InputArea.n_cols);
        unpooledError.fill(error(rowidx, colidx) / InputArea.n_elem);
        unpooledError %= pow(InputArea, normType - 1);
        unpooledError /= sum;
        output(arma::span(i, i + InputArea.n_rows - 1),
            arma::span(j, j + InputArea.n_cols - 1)) += unpooledError;
      }
    }
  }

  //! Locally-stored norm type.
  size_t normType;

  //! Locally-stored width of the pooling window.
  size_t kernelWidth;

  //! Locally-stored height of the pooling window.
  size_t kernelHeight;

  //! Locally-stored width of the stride operation.
  size_t strideWidth;

  //! Locally-stored height of the stride operation.
  size_t strideHeight;

  //! Rounding operation used.
  bool floor;

  //! Locally-stored number of input channels.
  size_t inSize;

  //! Locally-stored number of output channels.
  size_t outSize;

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! Locally-stored reset parameter used to initialize the module once.
  bool reset;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! Locally-stored output parameter.
  arma::cube outputTemp;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored transformed output parameter.
  arma::cube gTemp;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class LpPooling


} // namespace mlpack

// Include implementation.
#include "lp_pooling_impl.hpp"

#endif
