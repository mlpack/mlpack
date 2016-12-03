/**
 * @file constant.hpp
 * @author Marcus Edel
 *
 * Definition of the Constant class, which outputs a constant value given
 * any input.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONSTANT_HPP
#define MLPACK_METHODS_ANN_LAYER_CONSTANT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the constant layer. The constant layer outputs a given
 * constant value given any input value.
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
class Constant
{
 public:
  /**
   * Create the Constant object that outputs a given constant scalar value
   * given any input value.
   *
   * @param outSize The number of output units.
   * @param scalar The constant value used to create the constant output.
   */
  Constant(const size_t outSize, const double scalar) :
      inSize(0),
      outSize(outSize)
  {
    constantOutput = OutputDataType(outSize, 1);
    constantOutput.fill(scalar);
  }

  /**
   * Ordinary feed forward pass of a neural network. The forward pass fills the
   * output with the specified constant parameter.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType&& input, OutputType&& output)
  {
    if (inSize == 0)
    {
      inSize = input.n_elem;
    }

    output = constantOutput;
  }

  /**
   * Ordinary feed backward pass of a neural network. The backward pass of the
   * constant layer is returns always a zero output error matrix.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType&& /* input */, DataType&& /* gy */, DataType&& g)
  {
    g = arma::zeros<DataType>(inSize, 1);
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
    ar & data::CreateNVP(constantOutput, "constantOutput");
  }

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored constant output matrix.
  OutputDataType constantOutput;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class ConstantLayer

}; // namespace ann
}; // namespace mlpack

#endif