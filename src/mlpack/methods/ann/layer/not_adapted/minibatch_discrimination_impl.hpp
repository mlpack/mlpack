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

template<typename InputType, typename OutputType>
MiniBatchDiscrimination<InputType, OutputType
>::MiniBatchDiscrimination() :
  A(0),
  B(0),
  C(0),
  batchSize(0)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
MiniBatchDiscrimination<InputType, OutputType
>::MiniBatchDiscrimination(
    const size_t outSize,
    const size_t features) :
    a(0), // This will be set when OutputDimensions() is called.
    b(outSize - inSize),
    c(features),
    batchSize(0)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
void MiniBatchDiscrimination<InputType, OutputType>::SetWeights(
    typename OutputType::elem_type* weightsPtr)
{
  weights = OutputType(weightsPtr, b * c, a, false, false);
}

template<typename InputType, typename OutputType>
void MiniBatchDiscrimination<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  batchSize = input.n_cols;
  M = weight * input;
  arma::Cube<typename InputType::elem_type> cubeM(M.memptr(), b, c, batchSize,
      false, false);
  distances.set_size(b, batchSize, batchSize);

  for (size_t i = 0; i < cubeM.n_slices; ++i)
  {
    output.col(i).subvec(0, a - 1) = input.col(i);
    output.col(i).subvec(a, output.n_rows - 1).ones();
    for (size_t j = 0; j < cubeM.n_slices; ++j)
    {
      if (j < i)
      {
        output.col(i).subvec(a, output.n_rows - 1) += distances.slice(j).col(i);
      }
      else if (i == j)
      {
        continue;
      }
      else
      {
        distances.slice(i).col(j) =
            exp(-sum(abs(cubeM.slice(i) - cubeM.slice(j)), 1));
        output.col(i) += distances.slice(i).col(j);
      }
    }
  }
}

template<typename InputType, typename OutputType>
void MiniBatchDiscrimination<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  g = gy.head_rows(a);
  OutputType gM = gy.tail_rows(B);
  deltaM.zeros(b, c, batchSize);

  for (size_t i = 0; i < M.n_slices; ++i)
  {
    for (size_t j = 0; j < M.n_slices; ++j)
    {
      if (i == j)
      {
        continue;
      }
      InputType t = sign(M.slice(i) - M.slice(j));
      t.each_col() %=
          distances.slice(std::min(i, j)).col(std::max(i, j)) % gM.col(i);
      deltaM.slice(i) -= t;
      deltaM.slice(j) += t;
    }
  }

  OutputType deltaTemp(deltaM.memptr(), b * c, batchSize, false, true);
  g += weight.t() * deltaTemp;
}

template<typename InputType, typename OutputType>
void MiniBatchDiscrimination<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& /* error */,
    OutputType& gradient)
{
  gradient = vectorise(deltaTemp * input.t());
}

template<typename InputType, typename OutputType>
template<typename Archive>
void MiniBatchDiscrimination<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(a));
  ar(CEREAL_NVP(b));
  ar(CEREAL_NVP(c));
}

} // namespace mlpack

#endif
