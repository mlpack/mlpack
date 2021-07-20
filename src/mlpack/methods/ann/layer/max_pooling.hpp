/**
 * @file methods/ann/layer/max_pooling.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Definition of the MaxPooling class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MAX_POOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_MAX_POOLING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/*
 * The max pooling rule for convolution neural networks. Take the maximum value
 * within the receptive block.
 */
class MaxPoolingRule
{
 public:
  /*
   * Return the maximum value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.
   */
  template<typename MatType>
  size_t Pooling(const MatType& input)
  {
    return arma::as_scalar(arma::find(input.max() == input, 1));
  }
};

/**
 * Implementation of the MaxPooling layer.
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
class MaxPoolingType : public Layer<InputType, OutputType>
{
 public:
  //! Create the MaxPooling object.
  MaxPoolingType();

  /**
   * Create the MaxPooling object using the specified number of units.
   *
   * @param kernelWidth Width of the pooling window.
   * @param kernelHeight Height of the pooling window.
   * @param strideWidth Width of the stride operation.
   * @param strideHeight Width of the stride operation.
   * @param floor Rounding operator (floor or ceil).
   */
  MaxPoolingType(const size_t kernelWidth,
                 const size_t kernelHeight,
                 const size_t strideWidth = 1,
                 const size_t strideHeight = 1,
                 const bool floor = true);

  // TODO: copy constructor / move constructor

  MaxPoolingType* Clone() const { return new MaxPoolingType(*this); }

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

  //! Get the value of the rounding operation.
  bool const& Floor() const { return floor; }
  //! Modify the value of the rounding operation.
  bool& Floor() { return floor; }

  //! Get the size of the output.
  const std::vector<size_t>& OutputDimensions() const
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
   * @param poolingIndices The pooled indices.
   */
  void PoolingOperation(const InputType& input,
                        OutputType& output,
                        OutputType& poolingIndices)
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

        const size_t idx = pooling.Pooling(subInput);
        output(i, j) = subInput(idx);

        if (this->training)
        {
          arma::Mat<size_t> subIndices = indices(arma::span(rowidx,
              rowidx + kernelWidth - 1 - offset),
              arma::span(colidx, colidx + kernelHeight - 1 - offset));

          poolingIndices(i, j) = subIndices(idx);
        }
      }
    }
  }

  /**
   * Apply unpooling to the input and store the results.
   *
   * @param error The backward error.
   * @param output The pooled result.
   * @param poolingIndices The pooled indices.
   */
  void Unpooling(const InputType& error,
                 OutputType& output,
                 OutputType& poolingIndices)
  {
    for (size_t i = 0; i < poolingIndices.n_elem; ++i)
    {
      output(poolingIndices(i)) += error(i);
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

  //! Locally-stored number of channels.
  size_t channels;

  //! Locally-stored cached output dimensions.
  std::vector<size_t> outputDimensions;

  //! Locally-stored reset parameter used to initialize the module once.
  bool reset;

  //! Locally-stored stored rounding offset.
  size_t offset;

  //! Locally-stored number of input units.
  size_t batchSize;

  //! Locally-stored pooling strategy.
  MaxPoolingRule pooling;

  //! Locally-stored indices matrix parameter.
  arma::Mat<size_t> indices;

  //! Locally-stored indices column parameter.
  arma::Col<size_t> indicesCol;

  //! Locally-stored pooling indicies.
  std::vector<arma::Cube<typename InputType::elem_type>> poolingIndices;
}; // class MaxPoolingType

// Standard MaxPooling layer.
typedef MaxPoolingType<arma::mat, arma::mat> MaxPooling;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "max_pooling_impl.hpp"

#endif
