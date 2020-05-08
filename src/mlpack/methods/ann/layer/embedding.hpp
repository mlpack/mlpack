/**
 * @file embedding.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the Embedding class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_EMBEDDING_HPP
#define MLPACK_METHODS_ANN_LAYER_EMBEDDING_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Word Embeddings, a featurized word-level representation capable of capturing
 * the semantic meanings of words. It stores embeddings of a dictionary and can
 * be retreived using their indices. It can only be used as first layer in an
 * artificial neural network.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename InitializerType = RandomInitialization,
    typename RegularizerType = NoRegularizer
>
class Embedding
{
 public:
  /**
   * Create the Embedding object.
   */
  Embedding();

  /**
   * Create the Embedding layer object using specified parameters.
   *
   * @param dictionarySize The size of the dictionary i.e number of distinct
   *        words in the document.
   * @param embeddingDim The size of each embedding vector.
   * @param paddingIndex Whenever it encounters `paddingIndex`, it pads the
   *        output with embedding vector with zeros.
   * @param freeze Specifies whether to update weight matrix after each forward
   *        pass.
   * @param initializer The initialization rule for embedding matrix.
   */
  Embedding(const size_t dictionarySize,
            const size_t embeddingDim,
            const int paddingIndex = NULL,
            const bool freeze = false,
            const InitializerType initializer = RandomInitialization);

  /**
   * Reset the layer parameters.
   */
  void ResetParameters();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /*
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

  //! Get the value of dictionarySize.
  OutputDataType& DictionarySize() const { return dictionarySize; }
  //! Modify the dictionarySize.
  OutputDataType& DictionarySize() { return dictionarySize; }

  //! Get the value of embeddingDim.
  OutputDataType& EmbeddingDim() const { return embeddingDim; }
  //! Modify the embeddingDim.
  OutputDataType& EmbeddingDim() { return embeddingDim; }

  //! Get the value of paddingIndex.
  OutputDataType& PaddingIndex() const { return paddingIndex; }
  //! Modify the paddingIndex.
  OutputDataType& PaddingIndex() { return paddingIndex; }

  //! Get the parameters.
  OutputDataType& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the iutput parameter.
  OutputDataType& InputParameter() const { return inputParameter; }
  //! Modify the iutput parameter.
  OutputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  InputDataType& Delta() const { return delta; }
  //! Modify the delta.
  InputDataType& Delta() { return delta; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */);

 private:
  //! Locally-stored size of the vocabulary.
  size_t dictionarySize;

  //! Locally-stored size of each embedding vector.
  size_t embeddingDim;

  //! Locally-stored value of padding index.
  int paddingIndex;

  //! Specifies whether to update weight matrix after each forward pass.
  bool freeze;

  //! Locally-stored initialization rule.
  InitializerType initializer;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Embedding

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "embedding_impl.hpp"

#endif
