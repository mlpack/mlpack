/**
 * @file methods/ann/layer/mean_pooling.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Definition of the MeanPooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_HPP

#include <mlpack/prereqs.hpp>

#include "padding.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the MeanPooling.
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
class MeanPooling
{
 public:
  //! Create the MeanPooling object.
  MeanPooling();

  /**
   * Create the MeanPooling object using the specified number of units and padding.
   *
   * @param kernelWidth Width of the pooling window.
   * @param kernelHeight Height of the pooling window.
   * @param strideWidth Width of the stride operation.
   * @param strideHeight Width of the stride operation.
   * @param floor Rounding operator (true for floor, else ceil).
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   * @param paddingType The type of padding (Valid or Same). Defaults to None.
   */

  MeanPooling(const size_t kernelWidth,
              const size_t kernelHeight,
              const size_t strideWidth = 1,
              const size_t strideHeight = 1,
              const bool floor = true,
              const size_t inputWidth = 0,
              const size_t inputHeight = 0,
              const size_t padW = 0,
              const size_t padH = 0,
              const std::string paddingType = "None");
  /**
   * Create the MeanPooling object using the specified number of units and padding.
   *
   * @param kernelWidth Width of the pooling window.
   * @param kernelHeight Height of the pooling window.
   * @param strideWidth Width of the stride operation.
   * @param strideHeight Width of the stride operation.
   * @param floor Rounding operator (floor or ceil).
   * @param padW A two-value tuple indicating padding widths of the input.
   *             First value is padding at left side. Second value is padding on
   *             right side.
   * @param padH A two-value tuple indicating padding heights of the input.
   *             First value is padding at top. Second value is padding on
   *             bottom.
   * @param paddingType The type of padding (Valid or Same). Defaults to None.
   */
  MeanPooling(const size_t kernelWidth,
              const size_t kernelHeight,
              const size_t strideWidth,
              const size_t strideHeight,
              const bool floor,
              const size_t inputWidth,
              const size_t inputHeight,
              const std::tuple<size_t, size_t> padW,
              const std::tuple<size_t, size_t> padH,
              const std::string paddingType = "None");

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

  //! Get the value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

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
    for (size_t j = 0, colidx = 0; j < output.n_cols;
         ++j, colidx += strideHeight)
    {
      for (size_t i = 0, rowidx = 0; i < output.n_rows;
           ++i, rowidx += strideWidth)
      {
        arma::mat subInput = input(
            arma::span(rowidx, rowidx + kernelWidth - 1 - offset),
            arma::span(colidx, colidx + kernelHeight - 1 - offset));

        output(i, j) = arma::mean(arma::mean(subInput));
      }
    }
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
    const size_t rStep = input.n_rows / error.n_rows - offset;
    const size_t cStep = input.n_cols / error.n_cols - offset;

    arma::Mat<eT> unpooledError;
    for (size_t j = 0; j < input.n_cols - cStep; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows - rStep; i += rStep)
      {
        const arma::Mat<eT>& inputArea = input(arma::span(i, i + rStep - 1),
            arma::span(j, j + cStep - 1));

        const size_t w1 {i * output.n_rows / input.n_rows},
            w2 {(i + rStep - 1 - offset) * output.n_rows / input.n_rows};

        const size_t h1 {j * output.n_cols / input.n_cols},
            h2 {(j + cStep - 1 - offset) * output.n_cols / input.n_cols};

        unpooledError = arma::Mat<eT>(w2 - w1 + 1, h2 - h1 + 1);

        if (i / rStep < error.n_rows && j / cStep < error.n_cols)
        {
          unpooledError.fill(error(i / rStep, j / cStep) / inputArea.n_elem);
          output(arma::span(w1, w2 - offset), arma::span(h1, h2 - offset))
              += unpooledError;
        }
      }
    }
  }

  /*
   * Function to assign padding such that output size is same as input size.
   */
  void InitializeSamePadding();

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

  //! If true use maximum a posteriori during the forward pass.
  bool deterministic;

  //! Locally-stored stored rounding offset.
  size_t offset;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! If true the input is padded for the operation.
  bool isPadded;

  //! Locally-stored left-side padding width.
  size_t padWLeft;

  //! Locally-stored right-side padding width.
  size_t padWRight;

  //! Locally-stored bottom padding height.
  size_t padHBottom;

  //! Locally-stored top padding height.
  size_t padHTop;

  //! Locally-stored output parameter.
  arma::cube outputTemp;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored transformed padded input parameter.
  arma::cube inputPaddedTemp;

  //! Locally-stored transformed output parameter.
  arma::cube gTemp;

  //! Locally-stored padding layer.
  ann::Padding<> padding;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class MeanPooling


} // namespace ann
} // namespace mlpack

// Include implementation.
#include "mean_pooling_impl.hpp"

#endif
