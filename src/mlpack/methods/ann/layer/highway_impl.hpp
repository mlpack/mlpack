/**
 * @file highway_impl.hpp
 * @author Dhawal Arora
 *
 * Implementation of highway layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HIGHWAY_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_HIGHWAY_IMPL_HPP

// In case it hasn't yet been included.
#include "highway.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<
    typename StateGame, typename CarryGate,
    typename InputDataType, typename OutputDataType
>
Highway<StateGate, CarryGate,
        InputDataType, OutputDataType>::Highway(StateGate H, CarryGate C)
  : stateGate(H), carryGate(C)
{
  // Nothing to do here.
}

template<
    typename StateGame, typename CarryGate,
    typename InputDataType, typename OutputDataType
>
void Highway<StateGate, CarryGate,
             InputDataType, OutputDataType>::Forward(
    const InputType&& input, OutputType&& output)
{
  OutputDataType carry;
  carryGate.Forward(input, carry);
  OutputDataType eval;
  stateGate.Forward(input, eval);
  output = carry % input + (1. - carry) % eval;
}

template<
    typename StateGame, typename CarryGate,
    typename InputDataType, typename OutputDataType
>
template<typename DataType>
void Highway<StateGate, CarryGate,
             InputDataType, OutputDataType>::Backward(
    const DataType&& input, DataType&& gy, DataType&& g)
{
  DataType carryDerivative, evalDerivative;
  DataType error;
  DataType carry, eval;
  carryGate.Forward(input, carry);
  carryGate.Gradient(input, error, carryDerivative);
  stateGate.Gradient(input, error, evalDerivative);
  stateGate.Forward(input, eval);
  DataType derivative = carryDerivative % (input - eval) + evalDerivative % (1. - carry) + carry;
  g = gy % derivative;
}

template<
    typename StateGame, typename CarryGate,
    typename InputDataType, typename OutputDataType
>
template<typename Archive>
void Highway<StateGate, CarryGate,
        InputDataType, OutputDataType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  // Nothing to do here
  // TODO How to save two gates?
}

} // namespace ann
} // namespace mlpack

#endif
