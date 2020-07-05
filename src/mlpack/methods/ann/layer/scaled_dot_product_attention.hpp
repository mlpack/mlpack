/**
 * @file methods/ann/layer/scaled_dot_product_attention.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the ScaledDotProductAttention class.
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

#ifndef MLPACK_METHODS_ANN_LAYER_SCALED_DOT_PRODUCT_ATTENTION_HPP
#define MLPACK_METHODS_ANN_LAYER_SCALED_DOT_PRODUCT_ATTENTION_HPP

#include <mlpack/prereqs.hpp>
#include "layer_types.hpp"
#include "softmax.hpp"
#include "../init_rules/glorot_init.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class ScaledDotProductAttention
{
 public:
  //! Element Type of the input.
  typedef typename InputDataType::elem_type ElemType;

  /**
   * Default constructor.
   */
  ScaledDotProductAttention();

  /**
   * Create the ScaledDotProductAttention object using the specified parameters.
   *
   * @param tLen The length of target sequence.
   * @param sLen The length of the source sequence.
   * @param embedDim Total dimension of the model.
   * @param dropoutRate The dropout rate for attention output weights.
   * @param deterministic If false, dropout layer is omitted else dropout layer
   *        is applied with dropout rate `dropout`.
   */
  ScaledDotProductAttention(const size_t tLen,
    const size_t sLen,
    const size_t embedDim,
    const ElemType dropoutRate = 0.1,
    const bool deterministic = false);

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
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The actual input to the layer.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

  //! Get the two dimensional Attention Mask.
  InputDataType const& AttentionMask() const { return attnMask; }
  //! Modify the two dimensional Attention Mask.
  InputDataType& AttentionMask() { return attnMask; }

  //! Get Key Padding Mask.
  InputDataType const& KeyPaddingMask() const { return keyPaddingMask; }
  //! Modify the Key Padding Mask.
  InputDataType& KeyPaddingMask() { return keyPaddingMask; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the value of deterministic.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of deterministic.
  bool& Deterministic() { return deterministic; }

 private:
  //! Function to multiply two cube objects.
  template <typename eT = double>
  arma::Cube<eT> CubeMultiply(const arma::Cube<eT>& a,
                              const arma::Cube<eT>& b,
                              const bool aTranspose = false,
                              const bool bTranspose = false)
  {
    size_t rows = a.n_rows, cols = b.n_cols, slices = a.n_slices;
    Log::Assert(a.n_slices == b.n_slices);
    if (bTranspose)
    {
      Log::Assert(a.n_cols == b.n_cols);
      cols = b.n_rows;
    }
    else if (aTranspose)
    {
      Log::Assert(a.n_rows == b.n_cols);
      rows = a.n_cols;
    }
    else
      Log::Assert(a.n_cols == b.n_rows);

    arma::Cube<eT> z(rows, cols, slices);
    for (size_t i = 0; i < slices; ++i)
    {
      if (bTranspose)
        z.slice(i) = a.slice(i) * b.slice(i).t();
      else if (aTranspose)
        z.slice(i) = a.slice(i).t() * b.slice(i);
      else
        z.slice(i) = a.slice(i) * b.slice(i);
    }
    return z;
  }

  //! Locally-stored value of target sequence length.
  size_t tLen;

  //! Locally-stored value of source sequence length.
  size_t sLen;

  //! Locally-stored module output size.
  size_t embedDim;

  //! Locally-stored dropout rate used on output weights.
  ElemType dropoutRate;

  //! Whether the forward pass is deterministic.
  bool deterministic;

  //! Two dimensional Attention Mask.
  InputDataType attnMask;

  //! Key Padding Mask.
  InputDataType keyPaddingMask;

  //! Locally-stored attention output weight to be fed to last linear layer.
  arma::Cube<ElemType> attnOut;

  //! Softmax layer to represent the probabilities of next sequence.
  Softmax<InputDataType, OutputDataType> softmax;

  //! Locally-stored output of the softmax layer.
  arma::Cube<ElemType> softmaxOutput;

  //! Dropout layer (optional).
  Dropout<InputDataType, OutputDataType> dropout;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter.
  OutputDataType outputParameter;
}; // class ScalarDotProductAttention
} // namespace ann
} // namespace mlpack

// Include implementation.
#include "scaled_dot_product_attention_impl.hpp"

#endif
