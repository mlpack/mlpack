/**
 * @file methods/ann/layer/gru_impl.hpp
 * @author Sumedh Ghaisas
 * @author Zachary Ng
 *
 * Implementation of the GRU class, which implements a gru network
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
GRU<MatType>::GRU() :
    RecurrentLayer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
GRU<MatType>::GRU(const size_t outSize) :
    RecurrentLayer<MatType>(),
    inSize(0),
    outSize(outSize)
{
  // Nothing to do here.
}

template<typename MatType>
GRU<MatType>::GRU(const GRU& other) :
    RecurrentLayer<MatType>(other),
    inSize(other.inSize),
    outSize(other.outSize)
{
  // Nothing to do here.
}

template<typename MatType>
GRU<MatType>::GRU(GRU&& other) :
    RecurrentLayer<MatType>(std::move(other)),
    inSize(other.inSize),
    outSize(other.outSize)
{
  // Nothing to do here.
}

template<typename MatType>
GRU<MatType>& GRU<MatType>::operator=(const GRU& other)
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
GRU<MatType>& GRU<MatType>::operator=(GRU&& other)
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
void GRU<MatType>::SetWeights(const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, weightsIn.n_rows, weightsIn.n_cols);

  const size_t inputWeightSize = outSize * inSize;
  MakeAlias(resetGateWeight, weightsIn, outSize, inSize, 0);
  MakeAlias(updateGateWeight, weightsIn, outSize, inSize, inputWeightSize);
  MakeAlias(hiddenGateWeight, weightsIn, outSize, inSize, inputWeightSize * 2);

  const size_t recurrentWeightOffset = inputWeightSize * 3;
  const size_t recurrentWeightSize = outSize * outSize;
  MakeAlias(recurrentResetGateWeight, weightsIn, outSize, outSize,
      recurrentWeightOffset);
  MakeAlias(recurrentUpdateGateWeight, weightsIn, outSize, outSize,
      recurrentWeightOffset + recurrentWeightSize);
  MakeAlias(recurrentHiddenGateWeight, weightsIn, outSize, outSize,
      recurrentWeightOffset + recurrentWeightSize * 2);
}

template<typename MatType>
void GRU<MatType>::Forward(const MatType& input, MatType& output)
{
  // Convenience alias.
  const size_t batchSize = input.n_cols;

  // Set aliases from the recurrent state.
  MakeStateAliases(batchSize);

  // Compute internal state using the following algorithm.
  // r_t = sigmoid(W_r x_t + U_r y_{t - 1})
  // z_t = sigmoid(W_z x_t + U_z y_{t - 1})
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
  resetGate = 1 / (1 + exp(-resetGate));
  updateGate = 1 / (1 + exp(-updateGate));

  // Calculate candidate activation vector.
  hiddenGate = hiddenGateWeight * input;

  // Add recurrent portion to activation vector.
  if (this->HasPreviousStep())
  {
    hiddenGate += resetGate % (recurrentHiddenGateWeight * prevOutput);
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

  currentOutput = output;
}

template<typename MatType>
void GRU<MatType>::Backward(
    const MatType& /* input */,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  // Get aliases from the recurrent state.
  const size_t batchSize = output.n_cols;
  MakeStateAliases(batchSize);

  // Work backwards to get error at each gate.
  // y_t = (1 - z_t) % y_{t - 1} + z_t % h_t
  // dh_t = dy % z_t
  deltaHidden = gy % updateGate;
  // The hidden gate uses a tanh activation function.
  // The derivative of tanh(x) is actually 1 - tanh^2(x) but
  // tanh has already been applied to hiddenGate in Forward().
  deltaHidden = deltaHidden % (1 - square(hiddenGate));

  // y_t = (1 - z_t) % y_{t - 1} + z_t % h_t
  // dz_t = dy % h_t - dy % y_{t - 1}
  deltaUpdate = gy % hiddenGate;
  if (this->HasPreviousStep())
    deltaUpdate -= gy % prevOutput;
  // The reset and update gate use sigmoid activation.
  // The derivative is sigmoid(x) * (1 - sigmoid(x)).  Since sigmoid has
  // already been applied to the gates, it's just `x * (1 - x)`
  deltaUpdate = deltaUpdate % (updateGate % (1 - updateGate));

  if (this->HasPreviousStep())
  {
    // h_t = tanh(W_h x_t + r_t % (U_h y_{t - 1}))
    // dr_t = dh_t % (U_h y_{t - 1})
    deltaReset = deltaHidden % (recurrentHiddenGateWeight * prevOutput);
    deltaReset = deltaReset % (resetGate % (1 - resetGate));
  }
  else
  {
    deltaReset.zeros(deltaHidden.n_rows, deltaHidden.n_cols);
  }

  // Calculate the input error.
  // r_t = sigmoid(W_r x_t + U_r y_{t - 1})
  // z_t = sigmoid(W_z x_t + U_z y_{t - 1})
  // h_t =    tanh(W_h x_t + r_t % (U_h y_{t - 1}))
  // dx_t = W_r * dr_t + W_z * dz_t + W_h * dh_t
  g = resetGateWeight.t() * deltaReset +
      updateGateWeight.t() * deltaUpdate +
      hiddenGateWeight.t() * deltaHidden;
}

template<typename MatType>
void GRU<MatType>::Gradient(
    const MatType& input,
    const MatType& /* error */,
    MatType& gradient)
{
  // This implementation reuses the deltas from Backward() and assumes that
  // they have not been changed.

  size_t offset = 0;
  // Non recurrent reset gate weights.
  gradient.submat(offset, 0, offset + resetGateWeight.n_elem - 1, 0) =
      vectorise(deltaReset * input.t());
  offset += resetGateWeight.n_elem;
  // Non recurrent update gate weights.
  gradient.submat(offset, 0, offset + updateGateWeight.n_elem - 1, 0) =
      vectorise(deltaUpdate * input.t());
  offset += updateGateWeight.n_elem;
  // Non recurrent hidden gate weights.
  gradient.submat(offset, 0, offset + hiddenGateWeight.n_elem - 1, 0) =
      vectorise(deltaHidden * input.t());
  offset += hiddenGateWeight.n_elem;

  // nextDelta is not set until after the first step.
  if (!this->AtFinalStep())
  {
    // Recurrent reset gate weights.
    gradient.submat(offset, 0, offset + recurrentResetGateWeight.n_elem - 1,
        0) = vectorise(nextDeltaReset * currentOutput.t());
    offset += recurrentResetGateWeight.n_elem;
    // Recurrent update gate weights.
    gradient.submat(offset, 0, offset + recurrentUpdateGateWeight.n_elem - 1,
        0) = vectorise(nextDeltaUpdate * currentOutput.t());
    offset += recurrentUpdateGateWeight.n_elem;
    // Recurrent hidden gate weights.
    gradient.submat(offset, 0, offset + recurrentHiddenGateWeight.n_elem - 1,
        0) = vectorise(nextDeltaHidden * currentOutput.t());
    offset += recurrentHiddenGateWeight.n_elem;
  }

  // Move delta to nextDelta for the next step.
  nextDeltaReset = std::move(deltaReset);
  nextDeltaUpdate = std::move(deltaUpdate);
  nextDeltaHidden = std::move(deltaHidden);
}

template<typename MatType>
size_t GRU<MatType>::WeightSize() const
{
  return outSize * inSize * 3 + /* Input weight connections */
      outSize * outSize * 3; /* Recurrent weight connections */
}

template<typename MatType>
size_t GRU<MatType>::RecurrentSize() const
{
  // The recurrent state has to store the output, reset gate, update gate,
  // and hidden gate. The last 3 aren't recurrent but are stored in Forward()
  // and used in Backward()
  return outSize * 4;
}

template<typename MatType>
template<typename Archive>
void GRU<MatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<RecurrentLayer<MatType>>(this));

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
}

template<typename MatType>
void GRU<MatType>::MakeStateAliases(size_t batchSize)
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
