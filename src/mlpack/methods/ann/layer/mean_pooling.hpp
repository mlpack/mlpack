/**
 * @file methods/ann/layer/mean_pooling.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 * @author Shubham Agrawal
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
#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the MeanPooling.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *         computation.
 */
template <typename MatType = arma::mat>
class MeanPoolingType : public Layer<MatType>
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
   * @param floor If true, then a pooling operation that would oly part of the
   *              input will be skipped.
   */
  MeanPoolingType(const size_t kernelWidth,
                  const size_t kernelHeight,
                  const size_t strideWidth = 1,
                  const size_t strideHeight = 1,
                  const bool floor = true);

  // Virtual destructor.
  virtual ~MeanPoolingType() { }

  //! Copy the given MeanPoolingType.
  MeanPoolingType(const MeanPoolingType& other);
  //! Take ownership of the given MeanPoolingType.
  MeanPoolingType(MeanPoolingType&& other);
  //! Copy the given MeanPoolingType.
  MeanPoolingType& operator=(const MeanPoolingType& other);
  //! Take ownership of the given MeanPoolingType.
  MeanPoolingType& operator=(MeanPoolingType&& other);

  //! Clone the MeanPoolingType object. This handles polymorphism correctly.
  MeanPoolingType* Clone() const { return new MeanPoolingType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

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

  //! Compute the size of the output given `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Apply pooling to all slices of the input and store the results.
   *
   * @param input The input to be apply the pooling rule.
   * @param output The pooled result.
   */
  void PoolingOperation(
      const arma::Cube<typename MatType::elem_type>& input,
      arma::Cube<typename MatType::elem_type>& output);

  /**
   * Apply unpooling to the input and store the results.
   *
   * @param input The input to be apply the unpooling rule.
   * @param output The pooled result.
   */
  void Unpooling(const MatType& error, MatType& output);

  /**
   * Return the average value of the receptive block.
   *
   * @param input Input used to perform the pooling operation.  Could be an
   *     Armadillo subview.
   */
  typename MatType::elem_type Pooling(const MatType& input)
  {
    return arma::mean(vectorise(input));
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
}; // class MeanPoolingType

// Standard MeanPooling layer.
using MeanPooling = MeanPoolingType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "mean_pooling_impl.hpp"

#endif
