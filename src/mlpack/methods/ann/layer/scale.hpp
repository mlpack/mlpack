/**
 * @file methods/ann/layer/scale.hpp
 * @author Ryan Curtin
 *
 * Definition of the Scale layer, which multiplies its inputs by a constant
 * value.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SCALE_HPP
#define MLPACK_METHODS_ANN_LAYER_SCALE_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The Scale layer allows the input to be scaled by a fixed constant, whose
 * value is specified in the constructor.
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *     cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *     to also be in this type. The type also allows the computation and weight
 *     type to differ from the input type (Default: arma::mat).
 */
template<typename MatType = arma::mat>
class Scale : public Layer<MatType>
{
 public:
  // Convenience typedef to access the element type of the weights and data.
  using ElemType = typename MatType::elem_type;

  /**
   * Create the Scale layer with the specified parameter.
   *
   * @param scaleFactor Value to multiple the inputs by.
   */
  Scale(const ElemType scaleFactor = 1.0);

  //! Clone the CELU object. This handles polymorphism correctly.
  Scale* Clone() const { return new Scale(*this); }

  // Virtual destructor.
  virtual ~Scale() { }

  // Copy constructor.
  Scale(const Scale& other);

  // Move constructor.
  Scale(Scale&& other);

  // Copy assignment operator.
  Scale& operator=(const Scale& other);

  // Move assignement operator.
  Scale& operator=(Scale&& other);

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
   * f(x) by propagating x backwards through f. Using the results from the feed
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

  // Get the scale factor..
  const ElemType& ScaleFactor() const { return scaleFactor; }
  // Modify the scale factor.
  ElemType& ScaleFactor() { return scaleFactor; }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  // Factor to scale the inputs by.
  ElemType scaleFactor;
}; // class Scale

} // namespace mlpack

// Include implementation.
#include "scale_impl.hpp"

#endif
