/**
 * @file hard_tanh.hpp
 * @author Dhawal Arora
 *
 * Definition and implementation of the HardTanH layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HARD_TANH_HPP
#define MLPACK_METHODS_ANN_LAYER_HARD_TANH_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Hard Tanh activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *     max & : x > maxValue \\
 *     min & : x \le minValue \\
 *     x   & : otherwise
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     0 & : x > maxValue \\
 *     0 & : x \le minValue \\
 *     1 & : otherwise
 *   \end{array}
 * \right.
 * @f}
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
class HardTanH
{
 public:
  /**
   * Create the HardTanH object using the specified parameters. The range
   * of the linear region can be adjusted by specifying the maxValue and
   * minValue. Default (maxValue = 1, minValue = -1).
   *
   * @param maxValue Range of the linear region maximum value.
   * @param minValue Range of the linear region minimum value.
   */
  HardTanH(const double maxValue = 1, const double minValue = -1) :
      maxValue(maxValue), minValue(minValue)
  {
     // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType&& input, OutputType&& output)
  {
    output = input;
    for (size_t i = 0; i < input.n_elem; i++)
    {
      output(i) = (output(i) > maxValue ? maxValue :
          (output(i) < minValue ? minValue : output(i)));
    }
  }

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
  void Backward(const DataType&& input,
                DataType&& gy,
                DataType&& g)
  {
    g = gy;
    for (size_t i = 0; i < input.n_elem; i++)
    {
      if (input(i) < minValue || input(i) > maxValue)
      {
        g(i) = 0;
      }
    }
  }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the maximum value.
  double const& MaxValue() const { return maxValue; }
  //! Modify the maximum value.
  double& MaxValue() { return maxValue; }

  //! Get the minimum value.
  double const& MinValue() const { return minValue; }
  //! Modify the minimum value.
  double& MinValue() { return minValue; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(maxValue, "maxValue");
    ar & data::CreateNVP(minValue, "minValue");
  }

 private:
  /**
   * Computes the HardTanH function.
   *
   * @param x Input data.
   * @return f(x).
   */
  double Fn(const double x)
  {
    if (x > maxValue)
      return maxValue;
    else if (x < minValue)
      return minValue;
    return x;
  }

  /**
   * Computes the HardTanH function using a dense matrix as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */

  template<typename eT>
  void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    y = x;
    y.transform( [&](eT val) { return std::min(
        std::max( val, minValue ), maxValue ); } );
  }

  /**
   * Computes the first derivative of the HardTanH function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  double Deriv(const double x)
  {
    return (x > maxValue || x < minValue) ? 0 : 1;
  }

  /**
   * Computes the first derivative of the HardTanH function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputType, typename OutputType>
  void Deriv(const InputType&& x, OutputType& y)
  {
    y = x;

    for (size_t i = 0; i < x.n_elem; i++)
      y(i) = Deriv(x(i));
  }

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Maximum value for the HardTanH function.
  double maxValue;

  //! Minimum value for the HardTanH function.
  double minValue;
}; // class HardTanH

} // namespace ann
} // namespace mlpack

#endif
