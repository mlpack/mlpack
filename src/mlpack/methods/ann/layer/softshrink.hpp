/**
 * @file softshrink.hpp
 * @author Lakshya Ojha
 *
 * The soft shrinkage has thresholds proportional to the noise level given by 
 * the user.
 * The use of a soft-shrinkage activation function pro-vides adaptive denoising
 * at various noise levels using a single CNN(Convolution Neural) without a requirement to train a
 * unique CNN for each noise level.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SOFTSHRINK_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTSHRINK_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artifical Neural Network. */ {

/**
 * SoftShrinkage operator is defined as,
 *         | x - lambda, if x >  lambda
 *  f(x) = | x + lambda, if x < -lambda
 *         | 0, otherwise
 *
 * lambda is set to 0.5 by default.
 *
 *          | 1 , if x >  lambda
 *  f'(x) = | 1 , if x < -lambda
 *          | 0 , otherwise
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class SoftShrink
{
 public:
  /**
   * Create SoftShrink object using specified hyperparameter lambda.
   * Soft shrinkage has a threshold lambda which is calculated by multiplying the
   * noise level sigma of the input(noisy image) and a coefficient 'a' which is one
   * of the training parameter.
   * Default (lambda = 0.5). lambda should be > 0.
   * 
   * @param lambda parameter
   */
  SoftShrink(const double lambda = 0.5);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   * 
   * @param input Input data used for evaluating the SoftShrink function.
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
   * @param g The calculated gradient
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

  //! Get the hyperparameter lambda.
  double const& Lambda() const { return lambda; }
  //! Modify the hyperparameter lambda.
  double& Lambda() { return lambda; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   * Computes the value of Soft Shrink activation function.
   *
   * @param x Input data.
   * @return f(x) 
   */
  double Fn(const double x)
  {
    if (x > lambda)
    {
      return x - lambda;
    }
    else if (x < -1 * lambda)
    {
      return x + lambda;
    }
    return 0.;
  }

  /**
   * Computes the value of Soft Shrink activation function using a dense matrix
   * as input.
   * 
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    y.set_size(arma::size(x));
    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = Fn(x(i));
    }
  }

  /**
   * Computes the inverse of the Soft Shrink function
   *             | y + lambda, y > 0
   * f^{-1}(y) = | y - lambda, y < 0
   *             | 0, otherwise
   * @param y
   * @return f^{-1}(y)
   */
  double Inv(const double y)
  {
    if ( y > 0)
    {
      return y + lambda;
    }
    else if (y < 0)
    {
      return y - lambda;
    }
    return 0;
  }

  /**
   * Computes the inverse of the Soft Shrink function
   * 
   * @param y Input data.
   * @param x The resulting inverse of the input data
   */
  template<typename InputVecType, typename OutputVecType>
  void Inv(const InputVecType& y,
                  OutputVecType& x,
                  const double lambda = 0.5)
  {
    x.set_size(arma::size(y));
    for (size_t i = 0; i < y.n_elem; i++)
    {
      x(i) = Inv(y(i));
    }
  }

  /**
   * Computes the first derivate of the Soft Shrink function
   * 
   * @param y Input activation
   * @return f'(x) where f(x) = y
   */
  double Deriv(const double y)
  {
    if (y == 0)
      return 0;
    return 1;
  }

  /**
   * Computes the first derivative of the Soft Shrink function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives. Should be the matrix used to
   *          calculate activation y. 
   * @param lambda parameter, default value = 0.5
   */
  template<typename InputVecType, typename OutputVecType>
  void Deriv(const InputVecType& y,
                    OutputVecType& x)
  {
    x.set_size(arma::size(y));
    for (size_t i = 0; i < y.n_elem; i++)
    {
      x(i) = Deriv(y(i));
    }
  }

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Soft Shrink hyperparameter.
  double lambda;
}; // class SoftShrink

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "softshrink_impl.hpp"

#endif
