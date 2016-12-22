/**
 * @file multiply_constant_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the MultiplyConstantLayer class, which multiplies the input by
 * a (non-learnable) constant.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_LAYER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the multiply constant layer. The multiply constant layer
 * multiplies the input by a (non-learnable) constant.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class MultiplyConstantLayer
{
 public:
  /**
   * Create the BaseLayer object.
   */
  MultiplyConstantLayer(const double scalar) : scalar(scalar)
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network. Multiply the input with the
   * specified constant scalar value.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType& input, OutputType& output)
  {
    output = input * scalar;
  }

  /**
   * Ordinary feed backward pass of a neural network. The backward pass
   * multiplies the error with the specified constant scalar value.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType& /* input */, const DataType& gy, DataType& g)
  {
    g = gy * scalar;
  }

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(scalar, "scalar");
  }

 private:
  //! Locally-stored constant scalar value.
  const double scalar;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class MultiplyConstantLayer

}; // namespace ann
}; // namespace mlpack

#endif
