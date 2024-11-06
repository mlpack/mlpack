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
#include "layer.hpp"

namespace mlpack {

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
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class LookupType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the Lookup object using the specified vocabulary and embedding size.
   *
   * @param vocabSize The size of the vocabulary.
   * @param embeddingSize The length of each embedding vector.
   */
  LookupType(const size_t vocabSize = 0, const size_t embeddingSize = 0);

  //! Clone the LookupType object. This handles polymorphism correctly.
  LookupType* Clone() const { return new LookupType(*this); }

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

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the size of the vocabulary.
  size_t VocabSize() const { return vocabSize; }

  //! Get the length of each embedding vector.
  size_t EmbeddingSize() const { return embeddingSize; }

  //! Get the number of trainable parameters.
  const size_t WeightSize() const { return embeddingSize * vocabSize; }

  //! Get the dimensions of the output.  This layer adds an extra dimension for
  //! the embedding.
  const std::vector<size_t>& OutputDimensions() const
  {
    std::vector<size_t> result(inputDimensions.size() + 1, embeddingSize);
    for (size_t i = 0; i < inputDimensions.size(); ++i)
      result[i + 1] = inputDimensions[i];
    return result;
  }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored size of the vocabulary.
  size_t vocabSize;

  //! Locally-stored length of each embedding vector.
  size_t embeddingSize;

  //! Locally-stored weight object.
  OutputType weights;
}; // class Lookup

// Alias for using as embedding layer.
// template<typename MatType = arma::mat>
// using Embedding = Lookup<MatType, MatType>;
using Lookup = LookupType<arma::mat, arma::mat>;
using Embedding = LookupType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "lookup_impl.hpp"

#endif
