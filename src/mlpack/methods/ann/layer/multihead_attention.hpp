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
namespace ann /** Artificial Neural Network. */ {

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
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam RegularizerType Type of the regularizer to be used.
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class MultiheadAttention
{
 public:
  /**
   * Default constructor.
   */
  MultiheadAttention();

  /**
   * Create the MultiheadAttention object using the specified modules.
   *
   * @param tgtSeqLen Target sequence length.
   * @param srcSeqLen Source sequence length.
   * @param embedDim Total dimension of the model.
   * @param numHeads Number of parallel attention heads.
   */
  MultiheadAttention(const size_t tgtSeqLen,
                     const size_t srcSeqLen,
                     const size_t embedDim,
                     const size_t numHeads);

  /**
   * Reset the layer parameters.
   */
  void Reset();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input The query matrix.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
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
   * @param input The input data used for evaluating specified function.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& gradient);

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

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
  OutputDataType const& AttentionMask() const { return attnMask; }
  //! Modify the two dimensional Attention Mask.
  OutputDataType& AttentionMask() { return attnMask; }

  //! Get Key Padding Mask.
  OutputDataType const& KeyPaddingMask() const { return keyPaddingMask; }
  //! Modify the Key Padding Mask.
  OutputDataType& KeyPaddingMask() { return keyPaddingMask; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return grad; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return grad; }

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

 private:
  //! Element Type of the input.
  typedef typename OutputDataType::elem_type ElemType;

  //! Target sequence length.
  size_t tgtSeqLen;

  //! Source sequence lenght.
  size_t srcSeqLen;

  //! Locally-stored module output size.
  size_t embedDim;

  //! Locally-stored number of parallel attention heads.
  size_t numHeads;

  //! Dimensionality of each head.
  size_t headDim;

  //! Two dimensional Attention Mask of shape (tgtSeqLen, srcSeqLen).
  OutputDataType attnMask;

  //! Key Padding Mask.
  OutputDataType keyPaddingMask;

  //! Locally-stored weight matrix associated with query.
  OutputDataType queryWt;

  //! Locally-stored weight matrix associated with key.
  OutputDataType keyWt;

  //! Locally-stored weight matrix associated with value.
  OutputDataType valueWt;

  //! Locally-stored weight matrix associated with attnWt.
  OutputDataType outWt;

  //! Locally-stored bias associated with query.
  OutputDataType qBias;

  //! Locally-stored bias associated with key.
  OutputDataType kBias;

  //! Locall-stored bias associated with value.
  OutputDataType vBias;

  //! Locally-stored bias associated with attnWt.
  OutputDataType outBias;

  //! Locally-stored weights parameter.
  OutputDataType weights;

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
  Softmax<InputDataType, OutputDataType> softmax;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient.
  OutputDataType grad;

  //! Locally-stored output parameter.
  OutputDataType outputParameter;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class MultiheadAttention
} // namespace ann
} // namespace mlpack

// Include implementation.
#include "multihead_attention_impl.hpp"

#endif
