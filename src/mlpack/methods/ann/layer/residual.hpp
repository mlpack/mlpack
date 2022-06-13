/**
 * @file methods/ann/layer/residual.hpp
 * @author Shubham Agrawal
 *
 * Base class for neural network layers that are wrappers around other layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RESIDUAL_HPP
#define MLPACK_METHODS_ANN_LAYER_RESIDUAL_HPP

#include "../make_alias.hpp"
#include "multi_layer.hpp"

namespace mlpack {
namespace ann {

/**
 * A lambda "map-reduce" is a layer that is a wrapper around other layers.  
 * It passes the input through all of its child layers sequentially, returning
 * the output from reducing the output.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType>
class ResidualType : public MultiLayer<MatType>
{
 public:
  /**
   * Create an empty ResidualType that holds no layers of its own.  Be sure to add
   * layers with Add() before using!
   */
  ResidualType();

  //! Copy the given ResidualType.
  ResidualType(const ResidualType& other);
  //! Take ownership of the layers of the given ResidualType.
  ResidualType(ResidualType&& other);
  //! Copy the given ResidualType.
  ResidualType& operator=(const ResidualType& other);
  //! Take ownership of the given ResidualType.
  ResidualType& operator=(ResidualType&& other);

  //! Virtual destructor: delete all held layers.
  virtual ~ResidualType()
  {
    // Nothing to do here. 
  }

  //! Create a copy of the ResidualType (this is safe for polymorphic use).
  virtual ResidualType* Clone() const { return new ResidualType(*this); }

  /**
   * Perform a forward pass with the given input data.  `output` is expected to
   * have the correct size (e.g. number of rows equal to `OutputSize()` of the
   * last held layer; number of columns equal to `input.n_cols`).
   *
   * @param input Input data to pass through the ResidualType.
   * @param output Matrix to store output in.
   */
  virtual void Forward(const MatType& input, MatType& output);

  /**
   * Perform a backward pass with the given data.  `gy` is expected to be the
   * propagated error from the subsequent layer (or output), `input` is expected
   * to be the output from this layer when `Forward()` was called, and `g` will
   * store the propagated error from this layer (to be passed to the previous
   * layer as `gy`).
   *
   * It is expected that `g` has the correct size already (e.g., number of rows
   * equal to `OutputSize()` of the previous layer, and number of columns equal
   * to `input.n_cols`).
   *
   * This function is expected to be called for the same input data as
   * `Forward()` was just called for.
   *
   * @param input Output of Forward().
   * @param gy Propagated error from next layer.
   * @param g Matrix to store propagated error in for previous layer.
   */
  virtual void Backward(const MatType& input,
                        const MatType& gy,
                        MatType& g);

  /**
   * Compute the gradients of each layer.
   *
   * This function is expected to be called for the same input data as
   * `Forward()` and `Backward()` were just called for.  That is, `input` here
   * should be the same data as `Forward()` was called with.
   *
   * `gradient` is expected to have the correct size already (e.g., number of
   * rows equal to 1, and number of columns equal to `WeightSize()`).
   *
   * @param input Original input data provided to Forward().
   * @param error Error as computed by `Backward()`.
   * @param gradient Matrix to store the gradients in.
   */
  virtual void Gradient(const MatType& input,
                        const MatType& error,
                        MatType& gradient);

  /**
   * Compute the output dimensions of the ResidualType using `InputDimensions()`.
   * This computes the dimensions of each layer held by the ResidualType, and the
   * output dimensions are set to the output dimensions of the last layer.
   */
  virtual void ComputeOutputDimensions();

  //! Serialize the ResidualType.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
};

typedef ResidualType<arma::mat> Residual;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "residual_impl.hpp"

#endif
