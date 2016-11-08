/**
 * @file hard_tanh_layer.hpp
 * @author Dhawal Arora
 *
 * Definition and implementation of the HardTanHLayer layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HARD_TANH_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_HARD_TANH_LAYER_HPP

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
class HardTanHLayer
{
 public:
  /**
   * Create the HardTanHLayer object using the specified parameters. The range
   * of the linear region can be adjusted by specifying the maxValue and
   * minValue. Default (maxValue = 1, minValue = -1).
   *
   * @param maxValue Range of the linear region maximum value.
   * @param minValue Range of the linear region minimum value.
   */
  HardTanHLayer(const double maxValue = 1, const double minValue = -1) :
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
  void Forward(const InputType& input, OutputType& output)
  {
    Fn(input, output);
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
  void Backward(const DataType& input,
                const DataType& gy,
                DataType& g)
  {
    DataType derivative;
    Deriv(input, derivative);
    g = gy % derivative;
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
  template<typename eT>
  void Backward(const arma::Cube<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Cube<eT>& g)
  {
    // Generate a cube using the backpropagated error matrix.
    arma::Cube<eT> mappedError = arma::zeros<arma::cube>(input.n_rows,
        input.n_cols, input.n_slices);

    for (size_t s = 0, j = 0; s < mappedError.n_slices; s+= gy.n_cols, j++)
    {
      for (size_t i = 0; i < gy.n_cols; i++)
      {
        arma::Col<eT> temp = gy.col(i).subvec(
            j * input.n_rows * input.n_cols,
            (j + 1) * input.n_rows * input.n_cols - 1);

        mappedError.slice(s + i) = arma::Mat<eT>(temp.memptr(),
            input.n_rows, input.n_cols);
      }
    }

    arma::Cube<eT> derivative;
    Deriv(input, derivative);
    g = mappedError % derivative;
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
   * Computes the HardTanH function using a 3rd-order tensor as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  void Fn(const arma::Cube<eT>& x, arma::Cube<eT>& y)
  {
    y = x;
    for (size_t s = 0; s < x.n_slices; s++)
      Fn(x.slice(s), y.slice(s));
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
  void Deriv(const InputType& x, OutputType& y)
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
}; // class HardTanHLayer

} // namespace ann
} // namespace mlpack

#endif
