// Temporarily drop.
/**
 * @file methods/ann/layer/positional_encoding.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the Positional Encoding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_POSITIONAL_ENCODING_HPP
#define MLPACK_METHODS_ANN_LAYER_POSITIONAL_ENCODING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Positional Encoding injects some information about the relative or absolute
 * position of the tokens in the sequence.
 *
 * The input and the output have the same shape:
 * `(embedDim * maxSequenceLength, batchSize)`. The embeddings are stored
 * consequently.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class PositionalEncodingType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create PositionalEncodingType object.
   */
  PositionalEncodingType();

  /**
   * Create the PositionalEncoding layer object using the specified parameters.
   *
   * @param embedDim The length of the embedding vector.
   * @param maxSequenceLength Number of tokens in each sequence.
   */
  PositionalEncodingType(const size_t embedDim,
                         const size_t maxSequenceLength);

  //! Clone the PositionalEncodingType object. This handles polymorphism
  //! correctly.
  PositionalEncodingType* Clone() const
  {
    return new PositionalEncodingType(*this);
  }

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

  //! Get the positional encoding vector.
  InputType const& Encoding() const { return positionalEncoding; }

  size_t InputShape() const
  {
    return embedDim * maxSequenceLength;
  }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Initialize positional encodings for further use.
   */
  void InitPositionalEncoding();

  //! Locally-stored embedding dimension.
  size_t embedDim;

  //! Locally-stored maximum sequence length that has to be encoded.
  size_t maxSequenceLength;

  //! Locally-stored positional encodings.
  InputType positionalEncoding;
}; // class PositionalEncodingTest

// Standard PositionalEncoding layer.
using PositionalEncoding = PositionalEncodingType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "positional_encoding_impl.hpp"

#endif
