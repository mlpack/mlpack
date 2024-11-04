/**
 * @file methods/ann/layer/padding.hpp
 * @author Saksham Bansal
 *
 * Definition of the Padding class that pads the incoming data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PADDING_HPP
#define MLPACK_METHODS_ANN_LAYER_PADDING_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Padding module class. The Padding module applies
 * (zero-valued) padding on the input data.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class PaddingType : public Layer<MatType>
{
 public:
  /**
   * Create the Padding object using the specified number of output units.
   *
   * @param padWLeft Left padding width of the input.
   * @param padWRight Right padding width of the input.
   * @param padHTop Top padding height of the input.
   * @param padHBottom Bottom padding height of the input.
   */
  PaddingType(const size_t padWLeft = 0,
              const size_t padWRight = 0,
              const size_t padHTop = 0,
              const size_t padHBottom = 0);

  //! Clone the PaddingType object. This handles polymorphism correctly.
  PaddingType* Clone() const { return new PaddingType(*this); }

  //! Virtual destructor.
  virtual ~PaddingType() { }

  //! Copy the given PaddingType.
  PaddingType(const PaddingType& other);
  //! Take ownership of the given PaddingType.
  PaddingType(PaddingType&& other);
  //! Copy the given PaddingType.
  PaddingType& operator=(const PaddingType& other);
  //! Take ownership of the given PaddingType.
  PaddingType& operator=(PaddingType&& other);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  //! Get the left padding width.
  size_t PadWLeft() const { return padWLeft; }
  //! Modify the left padding width.
  size_t& PadWLeft() { return padWLeft; }

  //! Get the right padding width.
  size_t PadWRight() const { return padWRight; }
  //! Modify the right padding width.
  size_t& PadWRight() { return padWRight; }

  //! Get the top padding width.
  size_t PadHTop() const { return padHTop; }
  //! Modify the top padding width.
  size_t& PadHTop() { return padHTop; }

  //! Get the bottom padding width.
  size_t PadHBottom() const { return padHBottom; }
  //! Modify the bottom padding width.
  size_t& PadHBottom() { return padHBottom; }

  //! Compute the output dimensions of the layer using `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored left padding width.
  size_t padWLeft;

  //! Locally-stored right padding width.
  size_t padWRight;

  //! Locally-stored top padding height.
  size_t padHTop;

  //! Locally-stored bottom padding height.
  size_t padHBottom;

  //! Cached number of input maps.
  size_t totalInMaps;
}; // class PaddingType

// Standard Padding layer.
using Padding = PaddingType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "padding_impl.hpp"

#endif
