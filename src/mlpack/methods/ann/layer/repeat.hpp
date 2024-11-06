/**
 * @file methods/ann/layer/repeat.hpp
 * @author Adam Kropp
 *
 * Definition of the Repeat class, which repeats the input n times
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REPEAT_HPP
#define MLPACK_METHODS_ANN_LAYER_REPEAT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Repeat class. The Repeat class repeats the
 * input a specified number of times along a each dimension.  The output
 * will have the same number of dimensions as the input, with each dimension
 * multiplied by a specified multiple.  The input can be repeated in an
 * interleaved or block fashion, depending on the parameters given to the
 * constructor.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template <typename MatType = arma::mat>
class RepeatType : public Layer<MatType>
{
 public:
  //! Get Specific Col type, not only arma
  using UintCol = typename GetUColType<MatType>::type;
  using UintMat = typename GetUDenseMatType<MatType>::type;
  /**
   * Create the Repeat object.  Multiples will be empty (e.g. 1s for all
   * dimensions), so this is the equivalent of an Identity Layer.
   * Interleave will be false (e.g. repeat in blocks).
   */
  RepeatType();

  /**
   * Create the Repeat object, specifying the number of times to repeat
   * along each dimension, as well as whether to interleave the output or
   * repeat in blocks.
   *
   * @param multiples The number of times to repeat along each axis. Must be
   *        the same size or smaller than InputDimensions.
   * @apram interleave If true, the output will be interleaved (similar to
   *        arma::repelem).  If false, the output will be repeated in blocks.
   */
  RepeatType(std::vector<size_t> multiples, bool interleave = false);

  /**
   * Destroy the layers held by the model.
   */
  virtual ~RepeatType() { }

  //! Clone the RepeatType object. This handles polymorphism correctly.
  RepeatType* Clone() const override { return new RepeatType(*this); }

  //! Copy the given RepeatType layer.
  RepeatType(const RepeatType& other);
  //! Take ownership of the given RepeatType layer.
  RepeatType(RepeatType&& other) noexcept;
  //! Copy the given RepeatType layer.
  RepeatType& operator=(const RepeatType& other);
  //! Take ownership of the given RepeatType layer.
  RepeatType& operator=(RepeatType&& other) noexcept;

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output) override;

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param * (input) The input data (x) given to the forward pass.
   * @param * (output) The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g) override;

  //! Get the repeat multiples
  const std::vector<size_t>& Multiples() const { return multiples; }

  //! Get the repeat multiples for modification
  std::vector<size_t>& Multiples()
  {
    this->validOutputDimensions = false;
    return multiples;
  }

  //! Get the interleave parameter
  bool Interleave() const { return interleave; }

  //! Get the interleave parameter for modification
  bool& Interleave() { return interleave; }

  /**
   * @brief Computes the output dimensions of the Repeat layer.
   *
   * The ComputeOutputDimensions function computes the output dimensions of the
   * Repeat layer based on the input dimensions and the repeat multiples. The
   * output dimensions will have the same number of dimensions as the input,
   * with all dimensions other than the one specified in the axis being the same
   * size as the input.
   */
  void ComputeOutputDimensions() override;

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Parameter to indicate number of times to repeat along each dimension
  std::vector<size_t> multiples;

  //! Parameter to indicate whether to interleave the output
  bool interleave;

  // Cache the target indices for a single tensor for use
  // in the forward pass.
  UintCol outIdxs;

  // Cache the contributions of each output element to the
  // input elements for use in the backward pass.
  size_t sizeMult;
  UintMat backIdxs;
}; // class RepeatType.

// Standard Repeat layer.
using Repeat = RepeatType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "repeat_impl.hpp"

#endif
