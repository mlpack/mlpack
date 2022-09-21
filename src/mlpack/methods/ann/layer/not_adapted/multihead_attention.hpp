// Temporarily drop.
/**
 * @file methods/ann/layer/multihead_attention.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the MultiheadAttention class.
 *
 * @code
 * @article{NIPS'17,
 *   author  = {Ashish Vaswani, Llion Jones, Noam Shazeer, Niki Parmar,
 *              Aidan N. Gomez, Jakob Uszkoreit, Łukasz Kaiser,
 *              Illia Polosukhin},
 *   title   = {Attention Is All You Need},
 *   year    = {2017},
 *   url     = {http://arxiv.org/abs/1706.03762v5}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_MULTIHEAD_ATTENTION_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIHEAD_ATTENTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/softmax.hpp>
#include <mlpack/methods/ann/layer/dropout.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

namespace mlpack {

/**
 * Multihead Attention allows the model to jointly attend to information from
 * different representation subspaces at different positions. With a single
 * attention head, averaging inhibits this. [arxiv.org:1706.03762v5]
 *
 * The MultiheadAttention class takes concatenated form of query, key and value.
 * The query, key and value are concatenated into single matrix and fed to the
 * Forward function as input.
 *
 * The query, key and value are matrices of shapes
 * `(embedDim * tgtSeqLen, batchSize)`, `(embedDim * srcSeqLen, batchSize)`
 * and `(embedDim * srcSeqLen, batchSize)` respectively. The output is a matrix
 * of shape `(embedDim * tgtSeqLen, batchSize)`. The embeddings are stored
 * consequently.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam RegularizerType Type of the regularizer to be used.
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class MultiheadAttentionType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Default constructor.
   */
  MultiheadAttentionType();

  // TODO: does srcSeqLen need to be given?
  /**
   * Create the MultiheadAttention object using the specified modules.
   *
   * @param tgtSeqLen Target sequence length.
   * @param srcSeqLen Source sequence length.
   * @param embedDim Total dimension of the model.
   * @param numHeads Number of parallel attention heads.
   * @param attnMask Two dimensional Attention Mask.
   * @param keyPaddingMask Key Padding Mask.
   */
  MultiheadAttentionType(const size_t tgtSeqLen,
                         const size_t srcSeqLen,
                         const size_t embedDim,
                         const size_t numHeads,
                         const InputType& attnmask = InputType(),
                         const InputType& keyPaddingMask = InputType());

  //! Clone the MultiheadAttentionType object. This handles polymorphism
  //! correctly.
  MultiheadAttentionType* Clone() const
  {
    return new MultiheadAttentionType(*this);
  }

  /**
   * Reset the layer parameters.
   */
  void SetWeights(typename OutputType::elem_type* weightsPtr);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input The query matrix.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input data used for evaluating specified function.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the size of the weights.
  size_t WeightSize() const { return 4 * (embedDim + 1) * embedDim; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //! Get the target sequence length.
  size_t TgtSeqLen() const { return tgtSeqLen; }
  //! Modify the target sequence length.
  size_t& TgtSeqLen() { return tgtSeqLen; }

  //! Get the source sequence length.
  size_t SrcSeqLen() const { return srcSeqLen; }
  //! Modify the source sequence length.
  size_t& SrcSeqLen() { return srcSeqLen; }

  //! Get the embedding dimension.
  size_t EmbedDim() const { return embedDim; }
  //! Modify the embedding dimension.
  size_t& EmbedDim() { return embedDim; }

  //! Get the number of attention heads.
  size_t NumHeads() const { return numHeads; }
  //! Modify the number of attention heads.
  size_t& NumHeads() { return numHeads; }

  //! Get the two dimensional Attention Mask.
  OutputType const& AttentionMask() const { return attnMask; }
  //! Modify the two dimensional Attention Mask.
  OutputType& AttentionMask() { return attnMask; }

  //! Get Key Padding Mask.
  OutputType const& KeyPaddingMask() const { return keyPaddingMask; }
  //! Modify the Key Padding Mask.
  OutputType& KeyPaddingMask() { return keyPaddingMask; }

  const size_t WeightSize() const { return (4 * embedDim + 4) * embedDim; }

  const std::vector<size_t> OutputDimensions() const
  {
    // This returns the output as a 2-dimensional (embedDim * tgtSeqLen)
    // matrix.
    std::vector<size_t> outputDimensions(inputDimensions.size(), 1);
    outputDimensions[0] = embedDim;
    outputDimensions[1] = tgtSeqLen;

    return outputDimensions;
  }

  size_t InputShape() const
  {
    return embedDim * (tgtSeqLen + 2 * srcSeqLen);
  }

 private:
  //! Element Type of the output.
  typedef typename OutputType::elem_type ElemType;

  //! Target sequence length.
  size_t tgtSeqLen;

  //! Source sequence length.
  size_t srcSeqLen;

  //! Locally-stored dimensionality of each embedding vector.
  size_t embedDim;

  //! Locally-stored number of parallel attention heads.
  size_t numHeads;

  //! Dimensionality of each head.
  size_t headDim;

  //! Two dimensional Attention Mask of shape (tgtSeqLen, srcSeqLen).
  OutputType attnMask;

  //! Key Padding Mask.
  OutputType keyPaddingMask;

  //! Locally-stored weight matrix associated with query.
  OutputType queryWt;

  //! Locally-stored weight matrix associated with key.
  OutputType keyWt;

  //! Locally-stored weight matrix associated with value.
  OutputType valueWt;

  //! Locally-stored weight matrix associated with attnWt.
  OutputType outWt;

  //! Locally-stored bias associated with query.
  OutputType qBias;

  //! Locally-stored bias associated with key.
  OutputType kBias;

  //! Locall-stored bias associated with value.
  OutputType vBias;

  //! Locally-stored bias associated with attnWt.
  OutputType outBias;

  //! Locally-stored weights parameter.
  OutputType weights;

  //! Locally-stored projected query matrix over linear layer.
  arma::Cube<ElemType> qProj;

  //! Locally-stored projected key matrix over linear layer.
  arma::Cube<ElemType> kProj;

  //! Locally-stored projected value matrix over linear layer.
  arma::Cube<ElemType> vProj;

  //! Locally-stored result of output of dropout layer.
  arma::Cube<ElemType> scores;

  //! Locally-stored attention output weight to be fed to last linear layer.
  arma::Cube<ElemType> attnOut;

  //! Softmax layer to represent the probabilities of next sequence.
  Softmax softmax;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class MultiheadAttention

// Standard MultiheadAttention layer using no regularization.
typedef MultiheadAttentionType<arma::mat, arma::mat, NoRegularizer>
    MultiheadAttention;

} // namespace mlpack

// Include implementation.
#include "multihead_attention_impl.hpp"

#endif
