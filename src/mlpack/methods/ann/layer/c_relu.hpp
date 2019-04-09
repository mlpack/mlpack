/**
 * @file c_relu_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of CReLU layer.
 * Introduced by,
 * Wenling Shang, Kihyuk Sohn, Diogo Almeida, Honglak Lee,
 * "https://arxiv.org/abs/1603.05201", 16th March 2016.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_C_RELU_HPP
#define MLPACK_METHODS_ANN_LAYER_C_RELU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Concatenated ReLU has two outputs, one ReLU and one negative ReLU, concatenated together.
 * In other words, for positive x it produces [x, 0], and for negative x it produces [0, x].
 * Because it has two outputs, CReLU doubles the output dimension.
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
class CReLU
{
 public:
  /**
   * Create the CReLU object using the specified parameters.
   * The non zero gradient can be adjusted by specifying the parameter
   */
  CReLU();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   * Works only for 2D Tenosrs.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType&& input, OutputType&& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType&& input, DataType&& gy, DataType&& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */);

 private:
  /**
   * Computes the ReLU function
   *
   * @param x Input data.
   * @return f(x).
   */
  double Fn(const double x)
  {
    return std::max(x, 0.0);
  }

  /**
   * Computes the ReLU function using a dense matrix as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    y = arma::max(x, 0.0 * x);
  }

  /**
   * Computes the first derivative of the ReLU function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  double Deriv(const double x)
  {
    return (x >= 0) ? 1 : 0;
  }

  /**
   * Computes the first derivative of the ReLU function.
   *
   * @param x Input activations.
   * @param y The resulting derivatives.
   */

  template<typename InputType, typename OutputType>
  void Deriv(const InputType& x, OutputType& y)
  {
    y.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = Deriv(x(i));
    }
  }

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class CReLU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "c_relu_impl.hpp"

#endif
