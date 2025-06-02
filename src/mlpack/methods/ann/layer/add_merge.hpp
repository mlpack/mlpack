/**
 * @file methods/ann/layer/add_merge.hpp
 * @author Shubham Agrawal
 *
 * Definition of the AddMerge class, which acts as a addition container.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_MERGE_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_MERGE_HPP

#include "multi_layer.hpp"

namespace mlpack {

/**
 * Implementation of the AddMerge class. The AddMerge class works as a
 * feed-forward fully connected network container which plugs various layers
 * together.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType>
class AddMergeType : public MultiLayer<MatType>
{
 public:
  /**
   * Create an empty AddMergeType that holds no layers of its own.  Be sure to add
   * layers with Add() before using!
   */
  AddMergeType();

  //! Copy the given AddMergeType.
  AddMergeType(const AddMergeType& other);
  //! Take ownership of the layers of the given AddMergeType.
  AddMergeType(AddMergeType&& other);
  //! Copy the given AddMergeType.
  AddMergeType& operator=(const AddMergeType& other);
  //! Take ownership of the given AddMergeType.
  AddMergeType& operator=(AddMergeType&& other);

  //! Virtual destructor: delete all held layers.
  virtual ~AddMergeType()
  {
    // Nothing to do here.
  }

  //! Create a copy of the AddMergeType (this is safe for polymorphic use).
  AddMergeType* Clone() const { return new AddMergeType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& output,
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

  //! Compute the size of the output given `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Serialize the AddMergeType.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
};

using AddMerge = AddMergeType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "add_merge_impl.hpp"

#endif
