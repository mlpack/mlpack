/**
 * @file methods/ann/layer/noisylinear_impl.hpp
 * @author Nishant Kumar
 *
 * Implementation of the NoisyLinear layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NOISYLINEAR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_NOISYLINEAR_IMPL_HPP

// In case it hasn't yet been included.
#include "noisylinear.hpp"

namespace mlpack {

template<typename MatType>
NoisyLinearType<MatType>::NoisyLinearType(const size_t outSize) :
    Layer<MatType>(),
    outSize(outSize),
    inSize(0)
{
  // Nothing to do here.
}

template<typename MatType>
NoisyLinearType<MatType>::NoisyLinearType(const NoisyLinearType& other) :
    Layer<MatType>(other),
    outSize(other.outSize),
    inSize(other.inSize)
{
  // Nothing to do.
}

template<typename MatType>
NoisyLinearType<MatType>::NoisyLinearType(NoisyLinearType&& other) :
    Layer<MatType>(std::move(other)),
    outSize(std::move(other.outSize)),
    inSize(std::move(other.inSize))
{
  // Nothing to do.
}

template<typename MatType>
NoisyLinearType<MatType>&
NoisyLinearType<MatType>::operator=(const NoisyLinearType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    outSize = other.outSize;
    inSize = other.inSize;
  }

  return *this;
}

template<typename MatType>
NoisyLinearType<MatType>&
NoisyLinearType<MatType>::operator=(NoisyLinearType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    outSize = std::move(other.outSize);
    inSize = std::move(other.inSize);
  }

  return *this;
}

template<typename MatType>
void NoisyLinearType<MatType>::SetWeights(const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, 1, (outSize * inSize + outSize) * 2);
  MakeAlias(weightMu, weightsIn, outSize, inSize);
  MakeAlias(biasMu, weightsIn, outSize, 1, weightMu.n_elem);
  MakeAlias(weightSigma, weightsIn, outSize, inSize,
      weightMu.n_elem + biasMu.n_elem);
  MakeAlias(biasSigma, weightsIn, outSize, 1,
      weightMu.n_elem * 2 + biasMu.n_elem);

  this->ResetNoise();
}

template<typename MatType>
void NoisyLinearType<MatType>::ResetNoise()
{
  MatType epsilonIn;
  epsilonIn.randn(inSize, 1);
  epsilonIn = sign(epsilonIn) % sqrt(arma::abs(epsilonIn));

  MatType epsilonOut;
  epsilonOut.randn(outSize, 1);
  epsilonOut = sign(epsilonOut) % sqrt(arma::abs(epsilonOut));

  weightEpsilon = epsilonOut * epsilonIn.t();
  biasEpsilon = epsilonOut;
}

template<typename MatType>
void NoisyLinearType<MatType>::ResetParameters()
{
  const double muRange = 1 / std::sqrt(inSize);
  weightMu.randu();
  weightMu = muRange * (weightMu * 2 - 1);
  biasMu.randu();
  biasMu = muRange * (biasMu * 2 - 1);
  weightSigma.fill(0.5 / std::sqrt(inSize));
  biasSigma.fill(0.5 / std::sqrt(outSize));
}

template<typename MatType>
void NoisyLinearType<MatType>::Forward(const MatType& input, MatType& output)
{
  weight = weightMu + weightSigma % weightEpsilon;
  bias = biasMu + biasSigma % biasEpsilon;
  output = weight * input;
  output.each_col() += bias;
}

template<typename MatType>
void NoisyLinearType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = weight.t() * gy;
}

template<typename MatType>
void NoisyLinearType<MatType>::Gradient(
    const MatType& input, const MatType& error, MatType& gradient)
{
  // Locally stored to prevent multiplication twice.
  MatType weightGrad = error * input.t();

  // Gradients for mu values.
  gradient.rows(0, weight.n_elem - 1) = vectorise(weightGrad);
  gradient.rows(weight.n_elem, weight.n_elem + bias.n_elem - 1) = sum(error, 1);

  // Gradients for sigma values.
  gradient.rows(weight.n_elem + bias.n_elem, gradient.n_elem - bias.n_elem - 1)
      = vectorise(weightGrad % weightEpsilon);
  gradient.rows(gradient.n_elem - bias.n_elem, gradient.n_elem - 1)
      = sum(error, 1) % biasEpsilon;
}

template<typename MatType>
void NoisyLinearType<MatType>::ComputeOutputDimensions()
{
  inSize = this->inputDimensions[0];
  for (size_t i = 1; i < this->inputDimensions.size(); ++i)
      inSize *= this->inputDimensions[i];

  this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
      1);

  // The NoisyLinear layer flattens its output.
  this->outputDimensions[0] = outSize;
}

template<typename MatType>
template<typename Archive>
void NoisyLinearType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(inSize));

  if (cereal::is_loading<Archive>())
  {
    ResetNoise();
  }
}

} // namespace mlpack

#endif
