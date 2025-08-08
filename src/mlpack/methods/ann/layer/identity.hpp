/**
 * @file methods/ann/layer/identity.hpp
 * @author Shubham Agrawal
 *
 * Definition of the Identity class.
 *
 * This layer is used mainly in creating complex networks. This will be generally
 * used with AddMerge or Concat layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_IDENTITY_HPP
#define MLPACK_METHODS_ANN_LAYER_IDENTITY_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Identity layer.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *         computation.
 */
template <typename MatType = arma::mat>
class Identity : public Layer<MatType>
{
 public:
  // Convenience typedef to access the element type of the weights and data.
  using ElemType = typename MatType::elem_type;

  // Create the AdaptiveMaxPooling object.
  Identity();

  // Virtual destructor.
  virtual ~Identity()
  {
    // Nothing to do here.
  }

  //! Copy the given Identity.
  Identity(const Identity& other);
  //! Take ownership of the given Identity.
  Identity(Identity&& other);
  //! Copy the given Identity.
  Identity& operator=(const Identity& other);
  //! Take ownership of the given Identity.
  Identity& operator=(Identity&& other);

  //! Clone the Identity object.
  //! This handles polymorphism correctly.
  Identity* Clone() const
  {
    return new Identity(*this);
  }

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
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);
}; // class Identity

} // namespace mlpack

// Include implementation.
#include "identity_impl.hpp"

#endif
