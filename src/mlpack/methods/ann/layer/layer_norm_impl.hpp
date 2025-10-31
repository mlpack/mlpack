/**
 * @file methods/ann/layer/layer_norm_impl.hpp
 * @author Shikhar Jaiswal
 * @author Adam Kropp
 *
 * Implementation of the Layer Normalization class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_LAYERNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYERNORM_IMPL_HPP

// In case it is not included.
#include "layer_norm.hpp"

namespace mlpack {


template <typename MatType>
LayerNorm<MatType>::LayerNorm(const double eps) :
    eps(eps)
{
}

template<typename MatType>
void LayerNorm<MatType>::SetWeights(const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, 2 * size, 1);
  MakeAlias(gamma, weightsIn, size, 1);
  MakeAlias(beta, weightsIn, size, 1, gamma.n_elem);
}

template<typename MatType>
void LayerNorm<MatType>::CustomInitialize(
      MatType& W,
      const size_t elements)
{
  if (elements != 2 * size)
  {
    throw std::invalid_argument("LayerNorm::CustomInitialize(): wrong "
                                "elements size!");
  }
  MatType gammaTemp;
  MatType betaTemp;
  // Gamma acts as the scaling parameters for the normalized output.
  MakeAlias(gammaTemp, W, size, 1);
  // Beta acts as the shifting parameters for the normalized output.
  MakeAlias(betaTemp, W, size, 1, gammaTemp.n_elem);

  gammaTemp.ones();
  betaTemp.zeros();
}

template<typename MatType>
void LayerNorm<MatType>::Forward(
    const MatType& input, MatType& output)
{
  mean = arma::mean(input, 0);
  variance = arma::var(input, 1, 0);

  // Normalize the input.
  output = input.each_row() - mean;
  inputMean = output;
  output.each_row() /= sqrt(variance + ElemType(eps));

  // Reused in the backward and gradient step.
  normalized = output;

  // Scale and shift the output.
  output.each_col() %= gamma;
  output.each_col() += beta;
}

template<typename MatType>
void LayerNorm<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  const MatType stdInv = 1 / sqrt(variance + ElemType(eps));

  // dl / dxhat.
  const MatType norm = gy.each_col() % gamma;

  // sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
  const MatType var = -sum(norm % inputMean, 0) % pow(stdInv, ElemType(3)) / 2;

  // dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
  // dl / dmu * 1 / m.
  g = (norm.each_row() % stdInv) + (inputMean.each_row() %
      var * 2 / gy.n_rows);

  // sum (dl / dxhat * -1 / stdInv) + variance *
  // (sum -2 * (x - mu)) / m.
  g.each_row() += sum(norm.each_row() % -stdInv, 0) / gy.n_rows;
}

template<typename MatType>
void LayerNorm<MatType>::Gradient(
    const MatType& /* input */,
    const MatType& error,
    MatType& gradient)
{
  gradient.set_size(size + size, 1);

  // Step 5: dl / dy * xhat.
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = sum(normalized % error, 1);

  // Step 6: dl / dy.
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) = sum(error, 1);
}

template<typename MatType>
template<typename Archive>
void LayerNorm<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(eps));
}

} // namespace mlpack

#endif
