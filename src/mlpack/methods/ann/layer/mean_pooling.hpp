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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the MeanPooling.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class MeanPoolingType : public Layer<InputType, OutputType>
{
 public:
  //! Create the MeanPoolingType object.
  MeanPoolingType();

  /**
   * Create the MeanPooling object using the specified number of units.
   *
   * @param kernelWidth Width of the pooling window.
   * @param kernelHeight Height of the pooling window.
   * @param strideWidth Width of the stride operation.
   * @param strideHeight Width of the stride operation.
   * @param floor Set to true to use floor method.
   */
  MeanPoolingType(const size_t kernelWidth,
                  const size_t kernelHeight,
                  const size_t strideWidth = 1,
                  const size_t strideHeight = 1,
                  const bool floor = true);

  // TODO: copy constructor / move constructor
  MeanPoolingType* Clone() const { return new MeanPoolingType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  //! Get the kernel width.
  size_t const& KernelWidth() const { return kernelWidth; }
  //! Modify the kernel width.
  size_t& KernelWidth() { return kernelWidth; }

  //! Get the kernel height.
  size_t const& KernelHeight() const { return kernelHeight; }
  //! Modify the kernel height.
  size_t& KernelHeight() { return kernelHeight; }

  //! Get the stride width.
  size_t const& StrideWidth() const { return strideWidth; }
  //! Modify the stride width.
  size_t& StrideWidth() { return strideWidth; }

  //! Get the stride height.
  size_t const& StrideHeight() const { return strideHeight; }
  //! Modify the stride height.
  size_t& StrideHeight() { return strideHeight; }

  //! Get the value of the rounding operation
  bool const& Floor() const { return floor; }
  //! Modify the value of the rounding operation
  bool& Floor() { return floor; }

  //! Get the size of the output.
  const std::vector<size_t> OutputDimensions() const
  {
    outputDimensions = this->inputDimensions;

    // Compute the size of the output.
    if (floor)
    {
      outputDimensions[0] = std::floor((this->inputDimensions[0] -
          (double) kernelWidth) / (double) strideWidth + 1);
      outputDimensions[1] = std::floor((this->inputDimensions[1] -
          (double) kernelHeight) / (double) strideHeight + 1);
      offset = 0;
    }
    else
    {
      outputDimensions[0] = std::ceil((this->inputDimensions[0] -
          (double) kernelWidth) / (double) strideWidth + 1);
      outputDimensions[1] = std::ceil((this->inputDimensions[1] -
          (double) kernelHeight) / (double) strideHeight + 1);
      offset = 1;
    }

    // Higher dimensions are not modified.
    for (size_t i = 2; i < this->inputDimensions.size(); ++i)
      outputDimensions[i] = this->inputDimensions[i];

    // Cache input size and output size.
    channels = std::accumulate(this->inputDimensions.begin() + 2,
        this->inputDimensions.end(), 0);

    return outputDimensions;
  }

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
  void Pooling(const InputType& input, OutputType& output)
  {
    for (size_t j = 0, colidx = 0; j < output.n_cols;
         ++j, colidx += strideHeight)
    {
      for (size_t i = 0, rowidx = 0; i < output.n_rows;
           ++i, rowidx += strideWidth)
      {
        InputType subInput = input(
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
  void Unpooling(const InputType& input,
                 const OutputType& error,
                 OutputType& output)
  {
    const size_t rStep = input.n_rows / error.n_rows - offset;
    const size_t cStep = input.n_cols / error.n_cols - offset;

    OutputType unpooledError;
    for (size_t j = 0; j < input.n_cols - cStep; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows - rStep; i += rStep)
      {
        const InputType& inputArea = input(arma::span(i, i + rStep - 1),
            arma::span(j, j + cStep - 1));

        unpooledError = OutputType(inputArea.n_rows, inputArea.n_cols);
        unpooledError.fill(error(i / rStep, j / cStep) / inputArea.n_elem);

        output(arma::span(i, i + rStep - 1 - offset),
            arma::span(j, j + cStep - 1 - offset)) += unpooledError;
      }
    }
  }

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

  //! Locally-stored number channels.
  size_t channels;

  //! Locally-stored cached output dimensions.
  std::vector<size_t> outputDimensions;

  //! Locally-stored stored rounding offset.
  size_t offset;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! Cached last-seen input.
  arma::Cube<typename InputType::elem_type> inputTemp;
}; // class MeanPoolingType

// Standard MeanPooling layer.
typedef MeanPoolingType<arma::mat, arma::mat> MeanPooling;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "mean_pooling_impl.hpp"

#endif
