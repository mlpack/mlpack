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
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
NoisyLinearType<InputType, OutputType>::NoisyLinearType(const size_t outSize) :
    outSize(outSize)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void NoisyLinearType<InputType, OutputType>::SetWeights(
    typename OutputType::elem_type* weightsPtr)
{
  weights = OutputType(weightsPtr, 1, (outSize * inSize + outSize) * 2, false,
      true);

  weightMu = OutputType(weightsPtr, outSize, inSize, false, true);
  biasMu = OutputType(weightsPtr + weightMu.n_elem, outSize, 1, false, true);
  weightSigma = OutputType(weightsPtr + weightMu.n_elem + biasMu.n_elem,
      outSize, inSize, false, true);
  biasSigma = OutputType(weightsPtr + weightMu.n_elem * 2 + biasMu.n_elem,
      outSize, 1, false, true);

  this->ResetNoise();
}

template<typename InputType, typename OutputType>
void NoisyLinearType<InputType, OutputType>::ResetNoise()
{
  InputType epsilonIn = arma::randn<InputType>(inSize, 1);
  epsilonIn = arma::sign(epsilonIn) % arma::sqrt(arma::abs(epsilonIn));

  OutputType epsilonOut = arma::randn<OutputType>(outSize, 1);
  epsilonOut = arma::sign(epsilonOut) % arma::sqrt(arma::abs(epsilonOut));

  weightEpsilon = epsilonOut * epsilonIn.t();
  biasEpsilon = epsilonOut;
}

template<typename InputType, typename OutputType>
void NoisyLinearType<InputType, OutputType>::ResetParameters()
{
  const double muRange = 1 / std::sqrt(inSize);
  weightMu.randu();
  weightMu = muRange * (weightMu * 2 - 1);
  biasMu.randu();
  biasMu = muRange * (biasMu * 2 - 1);
  weightSigma.fill(0.5 / std::sqrt(inSize));
  biasSigma.fill(0.5 / std::sqrt(outSize));
}

template<typename InputType, typename OutputType>
void NoisyLinearType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  weight = weightMu + weightSigma % weightEpsilon;
  bias = biasMu + biasSigma % biasEpsilon;
  output = weight * input;
  output.each_col() += bias;
}

template<typename InputType, typename OutputType>
void NoisyLinearType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  g = weight.t() * gy;
}

template<typename InputType, typename OutputType>
void NoisyLinearType<InputType, OutputType>::Gradient(
    const InputType& input, const OutputType& error, OutputType& gradient)
{
  // Locally stored to prevent multiplication twice.
  OutputType weightGrad = error * input.t();

  // Gradients for mu values.
  gradient.rows(0, weight.n_elem - 1) = arma::vectorise(weightGrad);
  gradient.rows(weight.n_elem, weight.n_elem + bias.n_elem - 1)
      = arma::sum(error, 1);

  // Gradients for sigma values.
  gradient.rows(weight.n_elem + bias.n_elem, gradient.n_elem - bias.n_elem - 1)
      = arma::vectorise(weightGrad % weightEpsilon);
  gradient.rows(gradient.n_elem - bias.n_elem, gradient.n_elem - 1)
      = arma::sum(error, 1) % biasEpsilon;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void NoisyLinearType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));

  if (cereal::is_loading<Archive>())
  {
    ResetNoise();
  }
}

} // namespace ann
} // namespace mlpack

#endif
