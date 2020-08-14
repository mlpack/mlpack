/**
 * @file methods/ann/layer/elu_impl.hpp
 * @author Vivek Pal
 * @author Dakshit Agrawal
 *
 * Implementation of the ELU activation function as described by Djork-Arne
 * Clevert, Thomas Unterthiner and Sepp Hochreiter.
 *
 * Implementation of the SELU function as introduced by Klambauer et. al. in
 * Self Neural Networks.  The SELU activation function keeps the mean and
 * variance of the input invariant.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ELU_IMPL_HPP

// In case it hasn't yet been included.
#include "elu.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

// This constructor is called for SELU activation function.  The values of
// alpha and lambda are constant for normalized inputs.
template<typename InputDataType, typename OutputDataType>
ELU<InputDataType, OutputDataType>::ELU() :
    alpha(1.6732632423543774),
    lambda(1.0507009873554802),
    deterministic(false)
{
  // Nothing to do here.
}

// This constructor is called for ELU activation function.  The value of lambda
// is fixed and equal to 1.  'alpha' is a hyperparameter.
template<typename InputDataType, typename OutputDataType>
ELU<InputDataType, OutputDataType>::ELU(const double alpha) :
    alpha(alpha),
    lambda(1),
    deterministic(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void ELU<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output = arma::ones<OutputDataType>(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    if (input(i) < DBL_MAX)
    {
      output(i) = (input(i) > 0) ? lambda * input(i) : lambda *
          alpha * (std::exp(input(i)) - 1);
    }
  }

    if (!deterministic)
    {
      derivative.set_size(arma::size(input));
      for (size_t i = 0; i < input.n_elem; ++i)
      {
        derivative(i) = (input(i) > 0) ? lambda : output(i) +
            lambda * alpha;
      }
    }
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void ELU<InputDataType, OutputDataType>::Backward(
    const DataType& /* input */, const DataType& gy, DataType& g)
{
  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ELU<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(alpha);
  ar & CEREAL_NVP(lambda);
}

} // namespace ann
} // namespace mlpack

#endif
