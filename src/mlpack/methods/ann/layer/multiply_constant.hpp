/**
 * @file multiply_constant.hpp
 * @author Marcus Edel
 *
 * Definition of the MultiplyConstantLayer class, which multiplies the input by
 * a (non-learnable) constant.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_HPP

#include <mlpack/prereqs.hpp>

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
class MultiplyConstant
{
 public:
  /**
   * Create the MultiplyConstant object.
   */
  MultiplyConstant(const double scalar);

  /**
   * Ordinary feed forward pass of a neural network. Multiply the input with the
   * specified constant scalar value.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType&& input, OutputType&& output);

  /**
   * Ordinary feed backward pass of a neural network. The backward pass
   * multiplies the error with the specified constant scalar value.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType&& /* input */, DataType&& gy, DataType&& g);

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
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored constant scalar value.
  const double scalar;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class MultiplyConstant

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "multiply_constant_impl.hpp"

#endif
