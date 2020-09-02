/**
 * @file methods/ann/layer/lookup.hpp
 * @author Marcus Edel
 *
 * Definition of the Lookup class a particular convolution, where the width of
 * the convolution is 1.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LOOKUP_HPP
#define MLPACK_METHODS_ANN_LAYER_LOOKUP_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /* Artificial Neural Network. */ {

/**
 * The Lookup class stores word embeddings and retrieves them using tokens. The
 * Lookup layer is always the first layer of the network. The input to the
 * Lookup class is a matrix of shape (sequenceLength, batchSize). The matrix
 * consists of tokens which are used to lookup the table (i.e. weights) to find
 * the embeddings of those tokens.
 *
 * The input shape : (sequenceLength, batchSize).
 * The output shape : (embeddingSize, sequenceLength, batchSize).
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
class Lookup
{
 public:
  /**
   * Create the Lookup object using the specified vocabulary and embedding size.
   *
   * @param vocabSize The size of the vocabulary.
   * @param embeddingSize The length of each embedding vector.
   */
  Lookup(const size_t vocabSize = 0, const size_t embeddingSize = 0);

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

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& gradient);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! Get the size of the vocabulary.
  size_t VocabSize() const { return vocabSize; }

  //! Get the length of each embedding vector.
  size_t EmbeddingSize() const { return embeddingSize; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored size of the vocabulary.
  size_t vocabSize;

  //! Locally-stored length of each embedding vector.
  size_t embeddingSize;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Lookup

// Alias for using as embedding layer.
template<typename MatType = arma::mat>
using Embedding = Lookup<MatType, MatType>;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "lookup_impl.hpp"

#endif
