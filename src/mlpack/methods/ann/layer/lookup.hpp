/**
 * @file methods/ann/layer/lookup.hpp
 * @author Marcus Edel
 *
 * Definition of the Lookup (embedding) layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LOOKUP_HPP
#define MLPACK_METHODS_ANN_LAYER_LOOKUP_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

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
 * The output shape : (sequenceLength * embeddingSize, batchSize).
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<
    typename MatType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class Lookup : public Layer<MatType>
{
 public:
  using CubeType = typename GetCubeType<MatType>::type;

  //! Create the Lookup object.
  Lookup();

  /**
   * Create the Lookup layer object with the specified number of output
   * dimensions.
   *
   * @param outSize The output dimension.
   * @param regularizer The regularizer to use, optional (default: no
   *     regularizer).
   */
  Lookup(const size_t vocabSize = 0, const size_t embeddingSize = 0,
         RegularizerType regularizer = RegularizerType());

  virtual ~Lookup() { }

  //! Clone the Lookup object. This handles polymorphism correctly.
  Lookup* Clone() const { return new Lookup(*this); }

  //! Copy the other Lookup layer (but not weights).
  Lookup(const Lookup& layer);

  //! Take ownership of the members of the other Lookup layer (but not weights).
  Lookup(Lookup&& layer);

  //! Copy the other Lookup layer (but not weights).
  Lookup& operator=(const Lookup& layer);

  //! Take ownership of the members of the other Lookup layer (but not weights).
  Lookup& operator=(Lookup&& layer);

  /**
   * Reset the layer parameter (weights and bias). The method is called to
   * assign the allocated memory to the internal learnable parameters.
   */
  void SetWeights(const MatType& weightsIn);

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
   * @param * (input) The propagated input activation.
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
  const MatType& Parameters() const { return weights; }
  //! Modify the parameters.
  MatType& Parameters() { return weights; }

  //! Get the size of the weights.
  size_t WeightSize() const { return vocabSize * embeddingSize; }

  //! Compute the output dimensions of the layer given `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored size of the vocabulary.
  size_t vocabSize;

  //! Locally-stored length of each embedding vector.
  size_t embeddingSize;

  //! Locally-stored weight object.
  MatType weights;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class Lookup

} // namespace mlpack

// Include implementation.
#include "lookup_impl.hpp"

#endif
