/**
 * @file methods/ann/layer/positional_encoding.hpp
 * @author Kumar Utkarsh
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
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

#include "layer.hpp"

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
template<
    typename MatType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class PositionalEncoding : public Layer<MatType>
{
 public:
  // Convenience typedef to access the element type of the weights and data.
  using ElemType = typename MatType::elem_type;

  // Create the PositionalEncoding object.
  PositionalEncoding();

  /**
   * Create the PositionalEncoding layer object using the specified parameters.
   *
   * @param embedDim The length of the embedding vector.
   * @param maxSequenceLength Number of tokens in each sequence.
   */
  PositionalEncoding(const size_t embedDim,
                    const size_t maxSequenceLength,
                   RegularizerType regularizer = RegularizerType());

  //! Clone the PositionalEncoding object. This handles polymorphism correctly.
  PositionalEncoding* Clone() const { return new PositionalEncoding(*this); }

  //! Reset the layer parameter.
  void SetWeights(const MatType& weightsIn);

  //! Copy constructor.
  PositionalEncoding(const PositionalEncoding& layer);

  //! Move constructor.
  PositionalEncoding(PositionalEncoding&&);

  //! Copy assignment operator.
  PositionalEncoding& operator=(const PositionalEncoding& layer);

  //! Move assignment operator.
  PositionalEncoding& operator=(PositionalEncoding&& layer);

  //! Virtual destructor.
  virtual ~PositionalEncoding() { }

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

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient);

  //! Get the parameters.
  const MatType& Parameters() const { return positionalEncoding; }
  //! Modify the parameters.
  MatType& Parameters() { return positionalEncoding; }

  //! Get the number of weights in the layer.
  size_t WeightSize() const { return embedDim * maxSequenceLength; }

  //! Compute the output dimensions of the layer using `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Serialize the layer.
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
  MatType positionalEncoding;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class PositionalEncoding

} // namespace mlpack

// Include implementation.
#include "positional_encoding_impl.hpp"

#endif
