/**
 * @file methods/ann/layer/lstm_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the LSTM class, which implements a lstm network layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LSTM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LSTM_IMPL_HPP

// In case it hasn't yet been included.
#include "lstm.hpp"

namespace mlpack {

template<typename MatType>
LSTMType<MatType>::LSTMType() :
    RecurrentLayer<MatType>(),
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename MatType>
LSTMType<MatType>::LSTMType(const size_t outSize) :
    RecurrentLayer<MatType>(),
    inSize(0),
    outSize(outSize)
{
  // Nothing to do here.
}

template<typename MatType>
LSTMType<MatType>::LSTMType(const LSTMType& layer) :
    RecurrentLayer<MatType>(layer),
    inSize(layer.inSize),
    outSize(layer.outSize)
{
  // Nothing to do here.
}

template<typename MatType>
LSTMType<MatType>::LSTMType(LSTMType&& layer) :
    RecurrentLayer<MatType>(std::move(layer)),
    inSize(layer.inSize),
    outSize(layer.outSize)
{
  layer.inSize = 0;
  layer.outSize = 0;
}

template<typename MatType>
LSTMType<MatType>& LSTMType<MatType>::operator=(const LSTMType& layer)
{
  if (this != &layer)
  {
    RecurrentLayer<MatType>::operator=(layer);
    inSize = layer.inSize;
    outSize = layer.outSize;
  }

  return *this;
}

template<typename MatType>
LSTMType<MatType>& LSTMType<MatType>::operator=(LSTMType&& layer)
{
  if (this != &layer)
  {
    RecurrentLayer<MatType>::operator=(std::move(layer));
    inSize = layer.inSize;
    outSize = layer.outSize;

    layer.inSize = 0;
    layer.outSize = 0;
  }

  return *this;
}

template<typename MatType>
void LSTMType<MatType>::SetWeights(const MatType& weights)
{
  // Set the weight parameters for the inputs.
  const size_t inputWeightSize = outSize * inSize;
  MakeAlias(blockInputWeight, weights, outSize, inSize);
  MakeAlias(inputGateWeight, weights, outSize, inSize, inputWeightSize);
  MakeAlias(forgetGateWeight, weights, outSize, inSize, 2 * inputWeightSize);
  MakeAlias(outputGateWeight, weights, outSize, inSize, 3 * inputWeightSize);

  // Set the bias parameters for the inputs.
  const size_t biasOffset = 4 * inputWeightSize;
  MakeAlias(blockInputBias, weights, outSize, 1, biasOffset);
  MakeAlias(inputGateBias, weights, outSize, 1, biasOffset + outSize);
  MakeAlias(forgetGateBias, weights, outSize, 1, biasOffset + 2 * outSize);
  MakeAlias(outputGateBias, weights, outSize, 1, biasOffset + 3 * outSize);

  // Set the recurrent weight parameters.
  const size_t recurrentOffset = biasOffset + 4 * outSize;
  const size_t recurrentWeightSize = outSize * outSize;
  MakeAlias(recurrentBlockInputWeight, weights, outSize, outSize,
      recurrentOffset);
  MakeAlias(recurrentInputGateWeight, weights, outSize, outSize,
      recurrentOffset + recurrentWeightSize);
  MakeAlias(recurrentForgetGateWeight, weights, outSize, outSize,
      recurrentOffset + 2 * recurrentWeightSize);
  MakeAlias(recurrentOutputGateWeight, weights, outSize, outSize,
      recurrentOffset + 3 * recurrentWeightSize);

  // Set the peephole weight parameters.
  const size_t peepholeOffset = recurrentOffset + 4 * recurrentWeightSize;
  MakeAlias(peepholeInputGateWeight, weights, outSize, 1, peepholeOffset);
  MakeAlias(peepholeForgetGateWeight, weights, outSize, 1,
      peepholeOffset + outSize);
  MakeAlias(peepholeOutputGateWeight, weights, outSize, 1,
      peepholeOffset + 2 * outSize);
}

// Forward when cellState is not needed.
template<typename MatType>
void LSTMType<MatType>::Forward(const MatType& input, MatType& output)
{
  // Convenience alias.
  const size_t batchSize = input.n_cols;

  // The internal quantities are stored as recurrent state; so, set aliases
  // correctly for this time step.
  SetInternalAliases(batchSize);

  // Compute internal state:
  //
  // z_t  =    tanh(W_z x_t + R_z y_{t - 1} + b_z)
  // i_t  = sigmoid(W_i x_t + R_i y_{t - 1} + p_i % c_{t - 1} + b_i)
  // f_t  = sigmoid(W_f x_t + R_f y_{t - 1} + p_f % c_{t - 1} + b_f)
  // c_t  =         z_t % i_t + c_{t - 1} % f_t
  // o_t  = sigmoid(W_o x_t + R_o y_{t - 1} + p_o % c_t + b_o)
  // y_t  =    tanh(c_t) % o_t

  // Start by computing all non-recurrent portions.
  blockInput = blockInputWeight * input + repmat(blockInputBias, 1, batchSize);
  inputGate = inputGateWeight * input + repmat(inputGateBias, 1, batchSize);
  forgetGate = forgetGateWeight * input + repmat(forgetGateBias, 1, batchSize);
  outputGate = outputGateWeight * input + repmat(outputGateBias, 1, batchSize);

  // Now add in recurrent portions, if needed.
  if (this->HasPreviousStep())
  {
    blockInput += recurrentBlockInputWeight * prevRecurrent;
    inputGate += recurrentInputGateWeight * prevRecurrent +
        repmat(peepholeInputGateWeight, 1, batchSize) % prevCell;
    forgetGate += recurrentForgetGateWeight * prevRecurrent +
        repmat(peepholeForgetGateWeight, 1, batchSize) % prevCell;
  }

  // Apply nonlinearities.  (TODO: fast sigmoid?)
  blockInput = tanh(blockInput);
  inputGate = 1.0 / (1.0 + exp(-inputGate));
  forgetGate = 1.0 / (1.0 + exp(-forgetGate));

  // Compute the cell state.
  if (this->HasPreviousStep())
    thisCell = blockInput % inputGate + prevCell % forgetGate;
  else
    thisCell = blockInput % inputGate;

  // Now add recurrent portion to output gate.
  if (this->HasPreviousStep())
  {
    outputGate += recurrentOutputGateWeight * prevRecurrent +
        repmat(peepholeOutputGateWeight, 1, batchSize) % thisCell;
  }
  else
  {
    // If we don't have a previous step, we still have to consider the peephole
    // connection.
    outputGate += repmat(peepholeOutputGateWeight, 1, batchSize) % thisCell;
  }

  // Apply nonlinearity for output gate.
  outputGate = 1.0 / (1.0 + exp(-outputGate));

  // Finally, we can compute the output itself.
  output = tanh(thisCell) % outputGate;

  // If necessary, store the recurrent output.
  if (!this->AtFinalStep())
    thisRecurrent = output;
}

template<typename MatType>
void LSTMType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  // Compute backward partial derivatives, defined by the following equations.
  // Note that there is a little sleight of hand here between non-delta and
  // delta values.  `o_t`, for instance, refers to the output gate *after* the
  // nonlinearity was applied.  But `do_t` refers to the delta *before* the
  // nonlinearity.  In the paper, they use an overbar to highlight this
  // difference, but I don't have that luxury here in this code.
  //
  // dy_t = gy + R_z^T dz_{t + 1} + R_i^T di_{t + 1} + R_f^T df_{t + 1} +
  //             R_o^T do_{t + 1}
  //
  // do_t = dy_t % h(c_t) % (o_t % (1 - o_t))
  // dc_t = dy_t % o_t % (1 - c_t .^ 2) + p_o % do_t + p_i % di_{t + 1} +
  //                                      p_f % df_{t + 1} +
  //                                      dc_{t + 1} % f_{t + 1}
  //
  // df_t = dc_t % c_{t - 1} % (f_t % (1 - f_t))
  // di_t = dc_t % z_t       % (i_t % (1 - i_t))
  // dz_t = dc_t % i_t       % (1 - z_t .^ 2)
  //
  // dx_t = W_z^T dz_t + W_i^T di_t + W_f^T df_t + W_o^T do_t
  //
  // Before we start, set all the internal aliases, which will contain this time
  // step's values as computed in Forward().
  const size_t batchSize = output.n_cols;
  SetInternalAliases(batchSize);
  SetBackwardWorkspace(batchSize);

  // First attempt...
  if (this->AtFinalStep())
  {
    deltaY = gy;
  }
  else
  {
    deltaY = gy + recurrentBlockInputWeight.t() * nextDeltaBlockInput +
                  recurrentInputGateWeight.t() * nextDeltaInputGate +
                  recurrentForgetGateWeight.t() * nextDeltaForgetGate +
                  recurrentOutputGateWeight.t() * nextDeltaOutputGate;
  }

  deltaOutputGate = deltaY % tanh(thisCell) % (outputGate % (1.0 - outputGate));

  // Only first two terms if at final step
  if (this->AtFinalStep())
  {
    deltaCell = deltaY % outputGate % (1.0 - square(tanh(thisCell))) +
        repmat(peepholeOutputGateWeight, 1, batchSize) % deltaOutputGate;
  }
  else
  {
    // To update the cell state, we actually need to use the forget gate values
    // from the next time step.
    MatType nextForgetGate;
    MakeAlias(nextForgetGate, this->RecurrentState(this->CurrentStep() + 1),
        outSize, batchSize, 4 * outSize * batchSize);

    deltaCell = deltaY % outputGate % (1.0 - square(tanh(thisCell))) +
        repmat(peepholeOutputGateWeight, 1, batchSize) % deltaOutputGate +
        repmat(peepholeInputGateWeight, 1, batchSize) % nextDeltaInputGate +
        repmat(peepholeForgetGateWeight, 1, batchSize) % nextDeltaForgetGate +
        nextDeltaCell % nextForgetGate;
  }

  if (this->HasPreviousStep())
    deltaForgetGate = deltaCell % prevCell % (forgetGate % (1.0 - forgetGate));
  else
    deltaForgetGate.zeros();
  deltaInputGate = deltaCell % blockInput % (inputGate % (1.0 - inputGate));
  deltaBlockInput = deltaCell % inputGate % (1.0 - square(blockInput));

  // Finally, compute deltaX (which is what we wanted all along).
  g = blockInputWeight.t() * deltaBlockInput +
      inputGateWeight.t() * deltaInputGate +
      forgetGateWeight.t() * deltaForgetGate +
      outputGateWeight.t() * deltaOutputGate;

  // Save this alias for later.
  MakeAlias(thisY, output, output.n_rows, output.n_cols);
}

template<typename MatType>
void LSTMType<MatType>::Gradient(
    const MatType& input,
    const MatType& /* error */,
    MatType& gradient)
{
  // This implementation depends on Gradient() being called just after
  // Backward(), which is something we can safely assume.  So, the workspace
  // aliases are already set by SetBackwardWorkspace().
  //
  // In this implementation we won't use aliases; we'll just address the correct
  // part of the gradient directly.

  // dW_z = < dz_t, x_t >
  gradient.submat(0, 0, blockInputWeight.n_elem - 1, 0) =
      vectorise(deltaBlockInput * input.t());
  size_t offset = blockInputWeight.n_elem;
  // dW_i = < di_t, x_t >
  gradient.submat(offset, 0, offset + inputGateWeight.n_elem - 1, 0) =
      vectorise(deltaInputGate * input.t());
  offset += inputGateWeight.n_elem;
  // dW_f = < df_t, x_t >
  gradient.submat(offset, 0, offset + forgetGateWeight.n_elem - 1, 0) =
      vectorise(deltaForgetGate * input.t());
  offset += forgetGateWeight.n_elem;
  // dW_o = < do_t, x_t >
  gradient.submat(offset, 0, offset + outputGateWeight.n_elem - 1, 0) =
      vectorise(deltaOutputGate * input.t());
  offset += outputGateWeight.n_elem;

  // db_z = sum(dz_t)
  gradient.submat(offset, 0, offset + blockInputBias.n_elem - 1, 0) =
      sum(deltaBlockInput, 1);
  offset += blockInputBias.n_elem;
  // db_i = sum(di_t)
  gradient.submat(offset, 0, offset + inputGateBias.n_elem - 1, 0) =
      sum(deltaInputGate, 1);
  offset += inputGateBias.n_elem;
  // db_f = sum(df_t)
  gradient.submat(offset, 0, offset + forgetGateBias.n_elem - 1, 0) =
      sum(deltaForgetGate, 1);
  offset += forgetGateBias.n_elem;
  // db_o = sum(do_t)
  gradient.submat(offset, 0, offset + outputGateBias.n_elem - 1, 0) =
      sum(deltaOutputGate, 1);
  offset += outputGateBias.n_elem;

  // For the recurrent weights, the gradient does not apply at the first time
  // step.

  if (!this->AtFinalStep())
  {
    // dR_z = < dz_{t + 1}, y_t >
    gradient.submat(offset, 0,
                    offset + recurrentBlockInputWeight.n_elem - 1, 0) =
        vectorise(nextDeltaBlockInput * thisY.t());
    offset += recurrentBlockInputWeight.n_elem;

    // dR_i = < di_{t + 1}, y_t >
    gradient.submat(offset, 0,
                    offset + recurrentInputGateWeight.n_elem - 1, 0) =
        vectorise(nextDeltaInputGate * thisY.t());
    offset += recurrentInputGateWeight.n_elem;

    // dR_f = < df_{t + 1}, y_t >
    gradient.submat(offset, 0,
                    offset + recurrentForgetGateWeight.n_elem - 1, 0) =
        vectorise(nextDeltaForgetGate * thisY.t());
    offset += recurrentForgetGateWeight.n_elem;

    // dR_o = < do_{t + 1}, y_t>
    gradient.submat(offset, 0,
                    offset + recurrentOutputGateWeight.n_elem - 1, 0) =
        vectorise(nextDeltaOutputGate * thisY.t());
    offset += recurrentOutputGateWeight.n_elem;
  }
  else
  {
    offset += recurrentBlockInputWeight.n_elem +
        recurrentInputGateWeight.n_elem +
        recurrentForgetGateWeight.n_elem +
        recurrentOutputGateWeight.n_elem;
  }

  // Finally, the peephole connection gradients.

  // dp_i = c_t % di_{t + 1}
  if (!this->AtFinalStep())
  {
    gradient.submat(offset, 0,
                    offset + peepholeInputGateWeight.n_elem - 1, 0) =
        sum(thisCell % nextDeltaInputGate, 1);
  }
  offset += peepholeInputGateWeight.n_elem;

  // dp_f = c_t % df_{t + 1}
  if (!this->AtFinalStep())
  {
    gradient.submat(offset, 0,
                    offset + peepholeForgetGateWeight.n_elem - 1, 0) =
        sum(thisCell % nextDeltaForgetGate, 1);
  }
  offset += peepholeForgetGateWeight.n_elem;

  // dp_o = c_t % do_t
  gradient.submat(offset, 0,
                  offset + peepholeOutputGateWeight.n_elem - 1, 0) =
      sum(thisCell % deltaOutputGate, 1);
}

template<typename MatType>
size_t LSTMType<MatType>::WeightSize() const
{
  return 4 * inSize * outSize /* input weight connections */ +
      4 * outSize /* input bias */ +
      4 * outSize * outSize /* recurrent weight connections */ +
      3 * outSize /* peephole connections */;
}

template<typename MatType>
size_t LSTMType<MatType>::RecurrentSize() const
{
  // We have to account for the cell, recurrent connection, and the four
  // internal matrices: block input, input gate, forget gate, and output gate.
  // Technically those last four are not recurrent connections, but we use them
  // as 'stored state' that we compute in Forward() and then access in
  // Backward().
  return 6 * outSize;
}

template<typename MatType>
void LSTMType<MatType>::SetInternalAliases(const size_t batchSize)
{
  // Make all of the aliases for internal state point to the correct place.
  MatType& state = this->RecurrentState(this->CurrentStep());

  // First make aliases for the recurrent connections.
  MakeAlias(thisRecurrent, state, outSize, batchSize);
  MakeAlias(thisCell, state, outSize, batchSize, outSize * batchSize);

  // Now make aliases for the internal state members that we use as scratch
  // space for computation.
  MakeAlias(blockInput, state, outSize, batchSize, 2 * outSize * batchSize);
  MakeAlias(inputGate, state, outSize, batchSize, 3 * outSize * batchSize);
  MakeAlias(forgetGate, state, outSize, batchSize, 4 * outSize * batchSize);
  MakeAlias(outputGate, state, outSize, batchSize, 5 * outSize * batchSize);

  // Make aliases for the previous time step, too, if we can.
  if (this->HasPreviousStep())
  {
    MatType& prevState = this->RecurrentState(this->PreviousStep());

    MakeAlias(prevRecurrent, prevState, outSize, batchSize);
    MakeAlias(prevCell, prevState, outSize, batchSize, outSize * batchSize);
  }
}

template<typename MatType>
void LSTMType<MatType>::SetBackwardWorkspace(const size_t batchSize)
{
  // We need to hold enough space for two time steps.
  workspace.set_size(12 * outSize, batchSize);

  if (this->CurrentStep() % 2 == 0)
  {
    MakeAlias(deltaY, workspace, outSize, batchSize);
    MakeAlias(deltaBlockInput, workspace, outSize, batchSize,
        outSize * batchSize);
    MakeAlias(deltaInputGate, workspace, outSize, batchSize,
        2 * outSize * batchSize);
    MakeAlias(deltaForgetGate, workspace, outSize, batchSize,
        3 * outSize * batchSize);
    MakeAlias(deltaOutputGate, workspace, outSize, batchSize,
        4 * outSize * batchSize);
    MakeAlias(deltaCell, workspace, outSize, batchSize,
        5 * outSize * batchSize);

    MakeAlias(nextDeltaY, workspace, outSize, batchSize,
        6 * outSize * batchSize);
    MakeAlias(nextDeltaBlockInput, workspace, outSize, batchSize,
        7 * outSize * batchSize);
    MakeAlias(nextDeltaInputGate, workspace, outSize, batchSize,
        8 * outSize * batchSize);
    MakeAlias(nextDeltaForgetGate, workspace, outSize, batchSize,
        9 * outSize * batchSize);
    MakeAlias(nextDeltaOutputGate, workspace, outSize, batchSize,
        10 * outSize * batchSize);
    MakeAlias(nextDeltaCell, workspace, outSize, batchSize,
        11 * outSize * batchSize);
  }
  else
  {
    MakeAlias(nextDeltaY, workspace, outSize, batchSize);
    MakeAlias(nextDeltaBlockInput, workspace, outSize, batchSize,
        outSize * batchSize);
    MakeAlias(nextDeltaInputGate, workspace, outSize, batchSize,
        2 * outSize * batchSize);
    MakeAlias(nextDeltaForgetGate, workspace, outSize, batchSize,
        3 * outSize * batchSize);
    MakeAlias(nextDeltaOutputGate, workspace, outSize, batchSize,
        4 * outSize * batchSize);
    MakeAlias(nextDeltaCell, workspace, outSize, batchSize,
        5 * outSize * batchSize);

    MakeAlias(deltaY, workspace, outSize, batchSize,
        6 * outSize * batchSize);
    MakeAlias(deltaBlockInput, workspace, outSize, batchSize,
        7 * outSize * batchSize);
    MakeAlias(deltaInputGate, workspace, outSize, batchSize,
        8 * outSize * batchSize);
    MakeAlias(deltaForgetGate, workspace, outSize, batchSize,
        9 * outSize * batchSize);
    MakeAlias(deltaOutputGate, workspace, outSize, batchSize,
        10 * outSize * batchSize);
    MakeAlias(deltaCell, workspace, outSize, batchSize,
        11 * outSize * batchSize);
  }
}

template<typename MatType>
template<typename Archive>
void LSTMType<MatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<RecurrentLayer<MatType>>(this));

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));

  // Clear internal scratch space if we are loading.
  if (Archive::is_loading::value)
  {
    workspace.clear();

    deltaY.clear();
    deltaBlockInput.clear();
    deltaInputGate.clear();
    deltaForgetGate.clear();
    deltaOutputGate.clear();
    deltaCell.clear();

    nextDeltaY.clear();
    nextDeltaBlockInput.clear();
    nextDeltaInputGate.clear();
    nextDeltaForgetGate.clear();
    nextDeltaOutputGate.clear();
    nextDeltaCell.clear();
  }
}

} // namespace mlpack

#endif
