/**
 * @file methods/ann/layer/gru_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the GRUType class, which implements a gru network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GRU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_GRU_IMPL_HPP

// In case it hasn't yet been included.
#include "gru.hpp"

namespace mlpack {

template<typename MatType>
GRUType<MatType>::GRUType() :
    RecurrentLayer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
GRUType<MatType>::GRUType(const size_t outSize) :
    RecurrentLayer<MatType>(),
    inSize(0),
    outSize(outSize)
{
  // Nothing to do here.
}

template<typename MatType>
GRUType<MatType>::GRUType(const GRUType& other) :
    RecurrentLayer<MatType>(other),
    inSize(other.inSize),
    outSize(other.outSize)
{
  // Nothing to do here.
}

template<typename MatType>
GRUType<MatType>::GRUType(GRUType&& other) :
    RecurrentLayer<MatType>(std::move(other)),
    inSize(other.inSize),
    outSize(other.outSize)
{
  // Nothing to do here.
}

template<typename MatType>
GRUType<MatType>& GRUType<MatType>::operator=(const GRUType& other)
{
  if (this != &other)
  {
    RecurrentLayer<MatType>::operator=(other);
    inSize = other.inSize;
    outSize = other.outSize;
  }

  return *this;
}

template<typename MatType>
GRUType<MatType>& GRUType<MatType>::operator=(GRUType&& other)
{
  if (this != &other)
  {
    RecurrentLayer<MatType>::operator=(std::move(other));
    inSize = other.inSize;
    outSize = other.outSize;
  }

  return *this;
}


template<typename MatType>
void GRUType<MatType>::SetWeights(const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, weightsIn.n_rows, weightsIn.n_cols);

  const size_t inputWeightSize = outSize * inSize;
  MakeAlias(resetGateWeight, weightsIn, outSize, inSize, 0);
  MakeAlias(updateGateWeight, weightsIn, outSize, inSize, inputWeightSize);
  MakeAlias(hiddenGateWeight, weightsIn, outSize, inSize, inputWeightSize * 2);

  const size_t recurrentWeightOffset = inputWeightSize * 3;
  const size_t recurrentWeightSize = outSize * outSize;
  MakeAlias(recurrentResetGateWeight, weightsIn, outSize, inSize,
      recurrentWeightOffset);
  MakeAlias(recurrentUpdateGateWeight, weightsIn, outSize, inSize,
      recurrentWeightOffset + recurrentWeightSize);
  MakeAlias(recurrentHiddenGateWeight, weightsIn, outSize, inSize,
      recurrentWeightOffset + recurrentWeightSize * 2);
}

template<typename MatType>
void GRUType<MatType>::Forward(const MatType& input, MatType& output)
{
  // Convenience alias.
  const size_t batchSize = input.n_cols;

  // Set aliases from the recurrent state.
  MakeStateAliases(batchSize);

  // Compute internal state using the following algorithm.
  // z_t = sigmoid(W_z x_t + U_z y_{t - 1})
  // r_t = sigmoid(W_r x_t + U_r y_{t - 1})
  // h_t =    tanh(W_h x_t + r_t % (U_h y_{t - 1}))
  // y_t =        (1 - z_t) % y_{t - 1} + z_t % h_t

  // Process non recurrent input.
  updateGate = updateGateWeight * input;
  resetGate = resetGateWeight * input;

  // Add recurrent input.
  if (this->HasPreviousStep())
  {
    resetGate += recurrentResetGateWeight * prevOutput;
    updateGate += recurrentUpdateGateWeight * prevOutput;
  }

  // Apply sigmoid activation function.
  resetGate = 1.0 / (1.0 + exp(-resetGate));
  updateGate = 1.0 / (1.0 + exp(-updateGate));

  // Calculate canidate activation vector.
  hiddenGate = hiddenGateWeight * input;

  // Add recurrent portion to activation vector.
  if (this->HasPreviousStep())
  {
    hiddenGate += recurrentHiddenGateWeight * (resetGate % prevOutput);
  }

  // Apply tanh activation function.
  hiddenGate = tanh(hiddenGate);

  // Compute output.
  output = updateGate % hiddenGate;

  // Add recurrent portion to output.
  if (this->HasPreviousStep())
  {
    output += (1 - updateGate) % prevOutput;
  }

  // If necessary, store output in recurrent state.
  if (!this->AtFinalStep())
    currentOutput = output;
}

template<typename MatType>
void GRUType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  // Get aliases from the recurrent state.
  const size_t batchSize = output.n_cols;
  MakeStateAliases(batchSize);

  // Work backwards to get error at each gate
  MatType deltaPrev = gy % (-1 * updateGate);

  MatType deltaUpdate = gy % ((-1 * prevOutput) + hiddenGate);
  deltaUpdate = deltaUpdate % (1 - deltaUpdate);

  MatType deltaHidden = gy % updateGate;
  deltaHidden = 1 - square(tanh(deltaHidden));

  MatType deltaReset = recurrentHiddenGateWeight.t() * deltaHidden;
  deltaReset = deltaReset % (1 - deltaReset);

  g = resetGateWeight.t() * deltaReset +
      updateGateWeight.t() * deltaUpdate +
      hiddenGateWeight.t() * deltaHidden;
}

template<typename MatType>
void GRUType<MatType>::Gradient(
    const MatType& /* input */,
    const MatType& /* error */,
    MatType& gradient)
{
  // Compute gradient from partial derivatives.
}

template<typename MatType>
size_t GRUType<MatType>::WeightSize() const
{
  return outSize * inSize * 3 + /* Input weight connections */
      outSize * outSize * 3; /* Recurrent weight connections */
}

template<typename MatType>
size_t GRUType<MatType>::RecurrentSize() const
{
  // The recurrent state has to store the output, reset gate, update gate,
  // and hidden gate. The last 3 aren't recurrent but are stored in Forward()
  // and used in Backward()
  return outSize * 4;
}

template<typename MatType>
template<typename Archive>
void GRUType<MatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<RecurrentLayer<MatType>>(this));

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
}

template<typename MatType>
void GRUType<MatType>::MakeStateAliases(size_t batchSize)
{
  MatType& state = this->RecurrentState(this->CurrentStep());

  MakeAlias(currentOutput, state, outSize, batchSize);
  MakeAlias(resetGate, state, outSize, batchSize, outSize * batchSize);
  MakeAlias(updateGate, state, outSize, batchSize, 2 * outSize * batchSize);
  MakeAlias(hiddenGate, state, outSize, batchSize, 3 * outSize * batchSize);

  if (this->HasPreviousStep())
  {
    MatType& prevState = this->RecurrentState(this->PreviousStep());
    MakeAlias(prevOutput, prevState, outSize, batchSize);
  }
}

} // namespace mlpack

#endif
