/**
 * @file isru.hpp
 * @author Prince Gupta
 *
 * Definition of ISRU (Inverse Square Root Unit) activation function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ISRU_HPP
#define MLPACK_METHODS_ANN_LAYER_ISRU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artifical Neural Network. */ {

/**
 * The ISRU (Inverse Square Root Unit) activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& x / sqrt(1 + alpha* x ^ {2})
 * f'(x) &=& cube(1 / sqrt(1 + alpha * x ^ {2}))
 * @f
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class ISRU
{
 public:
  /**
   * Create ISRU object using specified hyperparameter alpha.
   * Default (alpha = 0.1). Alpha should be > 0.
   * ISRU is defined as f(x) = x / (1 + alpha * x^{2})^{1/2}.
   * 
   * @param alpha hyperparameter used to calculate ISRU function.
   */
  ISRU(const double alpha = 0.1);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   * 
   * @param input Input data used for evaluating the ISRU function.
   * @param output Resulting output activation
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType&& input, OutputType&& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   * 
   * @param input The propagated input activation f(x).
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType&& input,
                DataType&& gy,
                DataType&& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the hyperparameter alpha.
  double const& Alpha() const { return alpha; }
  //! Modify the hyperparameter alpha.
  double& Alpha() { return alpha; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   * Computes the value of ISRU activation function.
   *
   * @param x Input data.
   * @return f(x) 
   */
  double Fn(const double x)
  {
    return x / (std::sqrt(1 + alpha * std::pow(x, 2)));
  }

  /**
   * Computes the value of ISRU activation function using a dense matrix
   * as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    y = x / (arma::sqrt(1 + alpha * arma::pow(x, 2)));
  }

  /**
   * Computes the inverse of the ISRU function for a given input y.
   *
   * @param y Input data.
   * @return f^{-1}(y)
   */
  double Inv(const double y)
  {
    double yEdge = 1 / std::sqrt(alpha);
    if (y >= yEdge)
      return DBL_MAX;
    else if (y <= -yEdge)
      return -DBL_MAX;
    else
      return y / std::sqrt(1 - alpha * y * y);
  }

  /**
   * Computes the inverse of the ISRU function.
   *
   * @param y Input data.
   * @param x The resulting inverse of the input data.
   */
  template<typename InputVecType, typename OutputVecType>
  void Inv(const InputVecType& y, OutputVecType& x)
  {
    x.set_size(arma::size(y));

    for (size_t i = 0; i < x.n_elem; i++)
    {
      x(i) = Inv(y(i));
    }
  }

  /**
   * Computes the first derivate of the ISRU function.
   *
   * @param y Input activation.
   * @return f'(x) where f(x) = y.
   */
  double Deriv(const double y)
  {
    if (y == 0)
      return 1;
    return std::pow(y / Inv(y), 3);
  }

  /**
   * Computes the first derivative of the ISRU function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   * @param alpha parameter, default value = 0.1
   */
  template<typename InputVecType, typename OutputVecType>
  void Deriv(const InputVecType& y,
             OutputVecType& x)
  {
    Inv(y, x);
    for (size_t i = 0; i < x.n_elem; i++)
    {
      x(i) = Deriv(y(i));
    }
  }

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! ISRU hyperparameter
  double alpha;
}; // class ISRU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "isru_impl.hpp"

#endif
