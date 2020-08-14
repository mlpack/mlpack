/**
 * @file methods/ann/layer/minibatch_discrimination_impl.hpp
 * @author Saksham Bansal
 *
 * Implementation of the MiniBatchDiscrimination layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_IMPL_HPP

// In case it hasn't yet been included.
#include "minibatch_discrimination.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MiniBatchDiscrimination<InputDataType, OutputDataType
>::MiniBatchDiscrimination() :
  A(0),
  B(0),
  C(0),
  batchSize(0)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
MiniBatchDiscrimination<InputDataType, OutputDataType
>::MiniBatchDiscrimination(
    const size_t inSize,
    const size_t outSize,
    const size_t features) :
    A(inSize),
    B(outSize - inSize),
    C(features),
    batchSize(0)
{
  weights.set_size(A * B * C, 1);
}

template<typename InputDataType, typename OutputDataType>
void MiniBatchDiscrimination<InputDataType, OutputDataType>::Reset()
{
  weight = arma::mat(weights.memptr(), B * C, A, false, false);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MiniBatchDiscrimination<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  batchSize = input.n_cols;
  tempM = weight * input;
  M = arma::cube(tempM.memptr(), B, C, batchSize, false, false);
  distances.set_size(B, batchSize, batchSize);
  output.set_size(B, batchSize);

  for (size_t i = 0; i < M.n_slices; ++i)
  {
    output.col(i).ones();
    for (size_t j = 0; j < M.n_slices; ++j)
    {
      if (j < i)
      {
        output.col(i) += distances.slice(j).col(i);
      }
      else if (i == j)
      {
        continue;
      }
      else
      {
        distances.slice(i).col(j) =
          arma::exp(-arma::sum(abs(M.slice(i) - M.slice(j)), 1));
        output.col(i) += distances.slice(i).col(j);
      }
    }
  }

  output = join_cols(input, output); // (A + B) x batchSize
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MiniBatchDiscrimination<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  g = gy.head_rows(A);
  arma::Mat<eT> gM = gy.tail_rows(B);
  deltaM.zeros(B, C, batchSize);

  for (size_t i = 0; i < M.n_slices; ++i)
  {
    for (size_t j = 0; j < M.n_slices; ++j)
    {
      if (i == j)
      {
        continue;
      }
      arma::mat t = arma::sign(M.slice(i) - M.slice(j));
      t.each_col() %=
          distances.slice(std::min(i, j)).col(std::max(i, j)) % gM.col(i);
      deltaM.slice(i) -= t;
      deltaM.slice(j) += t;
    }
  }

  deltaTemp = arma::mat(deltaM.memptr(), B * C, batchSize, false, false);
  g += weight.t() * deltaTemp;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MiniBatchDiscrimination<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& /* error */,
    arma::Mat<eT>& gradient)
{
  gradient = arma::vectorise(deltaTemp * input.t());
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MiniBatchDiscrimination<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(A);
  ar & CEREAL_NVP(B);
  ar & CEREAL_NVP(C);

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (Archive::is_loading::value)
  {
    weights.set_size(A * B * C, 1);
  }
}

} // namespace ann
} // namespace mlpack

#endif
