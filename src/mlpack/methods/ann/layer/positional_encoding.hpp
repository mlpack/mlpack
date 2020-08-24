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
namespace ann /** Artificial Neural Network. */ {

/**
 * Positional Encoding injects some information about the relative or absolute
 * position of the tokens in the sequence.
 *
 * The input and the output have the same shape:
 * `(embedDim * maxSequenceLength, batchSize)`. The embeddings are stored
 * consequently.
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
class PositionalEncoding
{
 public:
  /**
   * Create PositionalEncoding object.
   */
  PositionalEncoding();

  /**
   * Create the PositionalEncoding layer object using the specified parameters.
   *
   * @param embedDim The length of the embedding vector.
   * @param maxSequenceLength Number of tokens in each sequence.
   */
  PositionalEncoding(const size_t embedDim,
                     const size_t maxSequenceLength);

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
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the positional encoding vector.
  InputDataType const& Encoding() const { return positionalEncoding; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

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
  InputDataType positionalEncoding;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class PositionalEncoding

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "positional_encoding_impl.hpp"

#endif
