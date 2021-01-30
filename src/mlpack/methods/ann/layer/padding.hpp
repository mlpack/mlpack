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
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Padding module class. The Padding module applies a bias term
 * to the incoming data.
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
class PaddingType : public Layer<InputType, OutputType>
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

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

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

  //! Locally-stored number of rows and columns of input.
  size_t nRows, nCols;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored output parameter object.
  OutputType outputParameter;
}; // class PaddingType

// Standard Padding layer.
typedef PaddingType<arma::mat, arma::mat> Padding;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "padding_impl.hpp"

#endif
