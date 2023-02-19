/**
 * @file methods/ann/layer/parametric_relu_impl.hpp
 * @author Prasanna Patil
 *
 * Definition of PReLU layer first introduced in the,
 * Kaiming He, Xiangyu Zhang, Shaoqing, Ren Jian Sun,
 * "Delving Deep into Rectifiers:
 * Surpassing Human-Level Performance on ImageNet Classification", 2014
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PRELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_PRELU_IMPL_HPP

// In case it hasn't yet been included.
#include "parametric_relu.hpp"

namespace mlpack {

template<typename MatType>
PReLUType<MatType>::PReLUType(const double userAlpha) :
    Layer<MatType>()
{
  alpha.set_size(1, 1);
  alpha(0) = userAlpha;
}

template<typename MatType>
PReLUType<MatType>::PReLUType(
    const PReLUType& other) :
    Layer<MatType>(other)
{
  userAlpha = other.userAlpha;
  alpha = other.alpha;
}

template<typename MatType>
PReLUType<MatType>::PReLUType(
    PReLUType&& other) :
    Layer<MatType>(std::move(other))
{
  userAlpha = std::move(other.userAlpha);
  alpha = std::move(other.alpha);
}

template<typename MatType>
PReLUType<MatType>&
PReLUType<MatType>::operator=(const PReLUType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    userAlpha = other.userAlpha;
    alpha = other.alpha;
  }

  return *this;
}

template<typename MatType>
PReLUType<MatType>&
PReLUType<MatType>::operator=(PReLUType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    userAlpha = std::move(other.userAlpha);
    alpha = std::move(other.alpha);
  }

  return *this;
}

template<typename MatType>
void PReLUType<MatType>::SetWeights(
    typename MatType::elem_type* weightsPtr)
{
  alpha = arma::mat(weightsPtr, 1, 1, false, false);

  //! Set value of alpha to the one given by user.
  // TODO: this doesn't even make any sense.  is it trainable or not?
  // why is there userAlpha?  is that for initialization only?
}

template<typename MatType>
void PReLUType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  // TODO: use transform()?
  output = input;
  if (this->training)
  {
    #pragma omp for
    for (size_t i = 0; i < input.n_elem; ++i)
      output(i) *= (input(i) >= 0) ? 1 : alpha(0);
  }
}

template<typename MatType>
void PReLUType<MatType>::Backward(
    const MatType& input, const MatType& gy, MatType& g)
{
  MatType derivative;
  derivative.set_size(arma::size(input));
  #pragma omp for
  for (size_t i = 0; i < input.n_elem; ++i)
    derivative(i) = (input(i) >= 0) ? 1 : alpha(0);

  g = gy % derivative;
}

template<typename MatType>
void PReLUType<MatType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  MatType zeros = arma::zeros<MatType>(input.n_rows, input.n_cols);
  gradient.set_size(1, 1);
  gradient(0) = arma::accu(error % arma::min(zeros, input)) / input.n_cols;
}

template<typename MatType>
template<typename Archive>
void PReLUType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(userAlpha));
  ar(CEREAL_NVP(alpha));
}

} // namespace mlpack

#endif
