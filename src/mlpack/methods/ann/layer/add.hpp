/**
 * @file methods/ann/layer/add.hpp
 * @author Marcus Edel
 *
 * Definition of the Add class that applies a bias term to the incoming data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Add layer. The Add module applies a bias term to the
 * incoming data.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType>
class AddType : public Layer<MatType>
{
 public:
  /**
   * Create the AddType object.  The output size of the layer will be the same
   * as the input size.
   */
  AddType();

  //! Clone the AddType object. This handles polymorphism correctly.
  AddType* Clone() const { return new AddType(*this); }

  // Virtual destructor.
  virtual ~AddType() { }

  //! Copy the given AddType layer.
  AddType(const AddType& other);
  //! Take ownership of the given AddType layer.
  AddType(AddType&& other);
  //! Copy the given AddType layer.
  AddType& operator=(const AddType& other);
  //! Take ownership of the given AddType layer.
  AddType& operator=(AddType&& other);

  /**
   * Forward pass: add the bias to the input.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Backward pass: send weights backwards (the bias does not affect anything).
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
   * Calculate the gradient using the output and the input activation.
   *
   * @param * (input) The propagated input.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& /* input */,
                const MatType& error,
                MatType& gradient);

  //! Return the weights of the network.
  const MatType& Parameters() const { return weights; }
  //! Modify the weights of the network.
  MatType& Parameters() { return weights; }

  //! Get the size of weights.
  size_t WeightSize() const { return outSize; }

  //! Compute the output dimensions of the layer, based on the internal values
  //! of `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Set the weights of the layer to use the given memory.
  void SetWeights(const MatType& weightsIn);

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  MatType weights;
}; // class Add

// Standard Add layer.
using Add = AddType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "add_impl.hpp"

#endif
