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
    outSize(0)
{
  // Nothing to do here.
}

template<typename MatType>
LSTMType<MatType>::LSTMType(const size_t outSize) :
    RecurrentLayer<MatType>(),
    outSize(outSize)
{
  // Nothing to do here.
}

template<typename MatType>
LSTMType<MatType>::LSTMType(const LSTMType& layer) :
    RecurrentLayer<MatType>(layer)
{
  // Nothing to do here.
}

template<typename MatType>
LSTMType<MatType>::LSTMType(LSTMType&& layer) :
    RecurrentLayer<MatType>(std::move(layer))
{
  // Nothing to do here.
}

template<typename MatType>
LSTMType<MatType>& LSTMType<MatType>::operator=(const LSTMType& layer)
{
  if (this != &layer)
  {
    RecurrentLayer<MatType>::operator=(layer);
  }

  return *this;
}

template<typename MatType>
LSTMType<MatType>& LSTMType<MatType>::operator=(LSTMType&& layer)
{
  if (this != &layer)
  {
    RecurrentLayer<MatType>::operator=(std::move(layer));
  }

  return *this;
}

template<typename MatType>
void LSTMType<MatType>::ClearRecurrentState(
    const size_t bpttSteps, const size_t batchSize)
{
  // Make sure all of the different matrices we will use to hold parameters are
  // at least as large as we need.
  inputGate.set_size(outSize, batchSize);
  forgetGate.set_size(outSize, batchSize);
  hiddenLayer.set_size(outSize, batchSize);
  outputGate.set_size(outSize, batchSize);

  inputGateActivation.set_size(outSize, batchSize, bpttSteps);
  forgetGateActivation.set_size(outSize, batchSize, bpttSteps);
  outputGateActivation.set_size(outSize, batchSize, bpttSteps);
  hiddenLayerActivation.set_size(outSize, batchSize, bpttSteps);

  cellActivation.set_size(outSize, batchSize, bpttSteps);
  outParameter.set_size(outSize, batchSize, bpttSteps);

  // Now reset recurrent values to 0.
  cell.zeros(outSize, batchSize, bpttSteps);
}

template<typename MatType>
void LSTMType<MatType>::SetWeights(const MatType& weights)
{
  // Set the weight parameter for the output gate.
  MakeAlias(input2GateOutputWeight, weights, outSize, inSize);
  size_t offset = input2GateOutputWeight.n_elem;
  MakeAlias(input2GateOutputBias, weights, outSize, 1, offset);
  offset += input2GateOutputBias.n_elem;

  // Set the weight parameter for the forget gate.
  MakeAlias(input2GateForgetWeight, weights, outSize, inSize, offset);
  offset += input2GateForgetWeight.n_elem;
  MakeAlias(input2GateForgetBias, weights, outSize, 1, offset);
  offset += input2GateForgetBias.n_elem;

  // Set the weight parameter for the input gate.
  MakeAlias(input2GateInputWeight, weights, outSize, inSize, offset);
  offset += input2GateInputWeight.n_elem;
  MakeAlias(input2GateInputBias, weights, outSize, 1, offset);
  offset += input2GateInputBias.n_elem;

  // Set the weight parameter for the hidden gate.
  MakeAlias(input2HiddenWeight, weights, outSize, inSize, offset);
  offset += input2HiddenWeight.n_elem;
  MakeAlias(input2HiddenBias, weights, outSize, 1, offset);
  offset += input2HiddenBias.n_elem;

  // Set the weight parameter for the output multiplication.
  MakeAlias(output2GateOutputWeight, weights, outSize, outSize, offset);
  offset += output2GateOutputWeight.n_elem;

  // Set the weight parameter for the output multiplication.
  MakeAlias(output2GateForgetWeight, weights, outSize, outSize, offset);
  offset += output2GateForgetWeight.n_elem;

  // Set the weight parameter for the input multiplication.
  MakeAlias(output2GateInputWeight, weights, outSize, outSize, offset);
  offset += output2GateInputWeight.n_elem;

  // Set the weight parameter for the hidden multiplication.
  MakeAlias(output2HiddenWeight, weights, outSize, outSize, offset);
  offset += output2HiddenWeight.n_elem;

  // Set the weight parameter for the cell multiplication.
  MakeAlias(cell2GateOutputWeight, weights, outSize, 1, offset);
  offset += cell2GateOutputWeight.n_elem;

  // Set the weight parameter for the cell - forget gate multiplication.
  MakeAlias(cell2GateForgetWeight, weights, outSize, 1, offset);
  offset += cell2GateOutputWeight.n_elem;

  // Set the weight parameter for the cell - input gate multiplication.
  MakeAlias(cell2GateInputWeight, weights, outSize, 1, offset);
}

// Forward when cellState is not needed.
template<typename MatType>
void LSTMType<MatType>::Forward(const MatType& input, MatType& output)
{
  // Convenience alias.
  const size_t batchSize = input.n_cols;

  inputGate = input2GateInputWeight * input;
  if (this->HasPreviousStep())
  {
    inputGate +=
        output2GateInputWeight * outParameter.slice(this->PreviousStep());
  }
  inputGate.each_col() += input2GateInputBias;

  forgetGate = input2GateForgetWeight * input;
  if (this->HasPreviousStep())
  {
    forgetGate += output2GateForgetWeight * outParameter.slice(
        this->PreviousStep());
  }
  forgetGate.each_col() += input2GateForgetBias;

  if (this->HasPreviousStep())
  {
    inputGate += repmat(cell2GateInputWeight, 1, batchSize) %
        cell.slice(this->PreviousStep());

    forgetGate += repmat(cell2GateForgetWeight, 1, batchSize) %
        cell.slice(this->PreviousStep());
  }

  inputGateActivation.slice(this->CurrentStep()) =
      1.0 / (1.0 + exp(-inputGate));
  forgetGateActivation.slice(this->CurrentStep()) =
      1.0 / (1.0 + exp(-forgetGate));

  hiddenLayer = input2HiddenWeight * input;
  if (this->HasPreviousStep())
  {
    hiddenLayer += output2HiddenWeight *
        outParameter.slice(this->PreviousStep());
  }
  hiddenLayer.each_col() += input2HiddenBias;

  hiddenLayerActivation.slice(this->CurrentStep()) = arma::tanh(hiddenLayer);

  if (!this->HasPreviousStep())
  {
    cell.slice(this->CurrentStep()) =
        inputGateActivation.slice(this->CurrentStep()) %
        hiddenLayerActivation.slice(this->CurrentStep());
  }
  else
  {
    cell.slice(this->CurrentStep()) =
        forgetGateActivation.slice(this->CurrentStep()) %
        cell.slice(this->PreviousStep()) +
        inputGateActivation.slice(this->CurrentStep()) %
        hiddenLayerActivation.slice(this->CurrentStep());
  }

  outputGate = input2GateOutputWeight * input +
      cell.slice(this->CurrentStep()).each_col() % cell2GateOutputWeight;
  if (this->HasPreviousStep())
  {
    outputGate +=
        output2GateOutputWeight * outParameter.slice(this->PreviousStep());
  }
  outputGate.each_col() += input2GateOutputBias;

  outputGateActivation.slice(this->CurrentStep()) =
      1.0 / (1.0 + exp(-outputGate));

  cellActivation.slice(this->CurrentStep()) =
      arma::tanh(cell.slice(this->CurrentStep()));

  // There's a bit of an issue here: we need to preserve the output for the next
  // time step, but we also need to set `output` to that.  Unfortunately for now
  // we make a copy, but it's possible that we could instead use an alias here,
  // or have `outParameter` hold a collection of aliases.
  outParameter.slice(this->CurrentStep()) =
      cellActivation.slice(this->CurrentStep()) %
      outputGateActivation.slice(this->CurrentStep());

  output = outParameter.slice(this->CurrentStep());
}

template<typename MatType>
void LSTMType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  MatType gyLocal;
  if (this->HasPreviousStep())
  {
    gyLocal = gy + output2GateOutputWeight.t() * outputGateError +
      output2GateForgetWeight.t() * forgetGateError +
      output2GateInputWeight.t() * inputGateError +
      output2HiddenWeight.t() * hiddenError;
  }
  else
  {
    // Make an alias.
    gyLocal = MatType(((MatType&) gy).memptr(), gy.n_rows, gy.n_cols,
        false, false);
  }

  outputGateError = gyLocal % cellActivation.slice(this->CurrentStep()) %
      (outputGateActivation.slice(this->CurrentStep()) %
      (1.0 - outputGateActivation.slice(this->CurrentStep())));

  MatType cellError = gyLocal %
      outputGateActivation.slice(this->CurrentStep()) %
      (1 - pow(cellActivation.slice(this->CurrentStep()), 2)) +
      outputGateError.each_col() % cell2GateOutputWeight;

  if (this->HasPreviousStep())
  {
    cellError += inputCellError;
  }

  if (this->HasPreviousStep())
  {
    forgetGateError = cell.slice(this->PreviousStep()) % cellError %
        (forgetGateActivation.slice(this->CurrentStep()) %
        (1.0 - forgetGateActivation.slice(this->CurrentStep())));
  }
  else
  {
    forgetGateError.zeros(forgetGateActivation.n_rows,
        forgetGateActivation.n_cols);
  }

  inputGateError = hiddenLayerActivation.slice(this->CurrentStep()) %
      cellError % (inputGateActivation.slice(this->CurrentStep()) %
      (1.0 - inputGateActivation.slice(this->CurrentStep())));

  hiddenError = inputGateActivation.slice(this->CurrentStep()) % cellError %
      (1 - pow(hiddenLayerActivation.slice(this->CurrentStep()), 2));

  inputCellError = forgetGateActivation.slice(this->CurrentStep()) % cellError +
      forgetGateError.each_col() % cell2GateForgetWeight +
      inputGateError.each_col() % cell2GateInputWeight;

  g = input2GateInputWeight.t() * inputGateError +
      input2HiddenWeight.t() * hiddenError +
      input2GateForgetWeight.t() * forgetGateError +
      input2GateOutputWeight.t() * outputGateError;
}

template<typename MatType>
void LSTMType<MatType>::Gradient(
    const MatType& input,
    const MatType& /* error */,
    MatType& gradient)
{
  // This implementation depends on Gradient() being called just after
  // Backward(), which is something we can safely assume.

  // Input2GateOutputWeight and input2GateOutputBias gradients.
  gradient.submat(0, 0, input2GateOutputWeight.n_elem - 1, 0) =
      vectorise(outputGateError * input.t());
  gradient.submat(input2GateOutputWeight.n_elem, 0,
      input2GateOutputWeight.n_elem + input2GateOutputBias.n_elem - 1, 0) =
      sum(outputGateError, 1);
  size_t offset = input2GateOutputWeight.n_elem + input2GateOutputBias.n_elem;

  // input2GateForgetWeight and input2GateForgetBias gradients.
  gradient.submat(offset, 0, offset + input2GateForgetWeight.n_elem - 1, 0) =
      vectorise(forgetGateError * input.t());
  gradient.submat(offset + input2GateForgetWeight.n_elem, 0,
      offset + input2GateForgetWeight.n_elem +
      input2GateForgetBias.n_elem - 1, 0) = sum(forgetGateError, 1);
  offset += input2GateForgetWeight.n_elem + input2GateForgetBias.n_elem;

  // input2GateInputWeight and input2GateInputBias gradients.
  gradient.submat(offset, 0, offset + input2GateInputWeight.n_elem - 1, 0) =
      vectorise(inputGateError * input.t());
  gradient.submat(offset + input2GateInputWeight.n_elem, 0,
      offset + input2GateInputWeight.n_elem +
      input2GateInputBias.n_elem - 1, 0) = sum(inputGateError, 1);
  offset += input2GateInputWeight.n_elem + input2GateInputBias.n_elem;

  // input2HiddenWeight and input2HiddenBias gradients.
  gradient.submat(offset, 0, offset + input2HiddenWeight.n_elem - 1, 0) =
      vectorise(hiddenError * input.t());
  gradient.submat(offset + input2HiddenWeight.n_elem, 0,
      offset + input2HiddenWeight.n_elem + input2HiddenBias.n_elem - 1, 0) =
      sum(hiddenError, 1);
  offset += input2HiddenWeight.n_elem + input2HiddenBias.n_elem;

  // output2GateOutputWeight gradients.
  gradient.submat(offset, 0, offset + output2GateOutputWeight.n_elem - 1, 0) =
      vectorise(outputGateError * outParameter.slice(this->CurrentStep()).t());
  offset += output2GateOutputWeight.n_elem;

  // output2GateForgetWeight gradients.
  gradient.submat(offset, 0, offset + output2GateForgetWeight.n_elem - 1, 0) =
      vectorise(forgetGateError * outParameter.slice(this->CurrentStep()).t());
  offset += output2GateForgetWeight.n_elem;

  // output2GateInputWeight gradients.
  gradient.submat(offset, 0, offset + output2GateInputWeight.n_elem - 1, 0) =
      vectorise(inputGateError * outParameter.slice(this->CurrentStep()).t());
  offset += output2GateInputWeight.n_elem;

  // output2HiddenWeight gradients.
  gradient.submat(offset, 0, offset + output2HiddenWeight.n_elem - 1, 0) =
      vectorise(hiddenError * outParameter.slice(this->CurrentStep()).t());
  offset += output2HiddenWeight.n_elem;

  // cell2GateOutputWeight gradients.
  gradient.submat(offset, 0, offset + cell2GateOutputWeight.n_elem - 1, 0) =
      sum(outputGateError % cell.slice(this->CurrentStep()), 1);
  offset += cell2GateOutputWeight.n_elem;

  // cell2GateForgetWeight and cell2GateInputWeight gradients.
  if (this->HasPreviousStep())
  {
    gradient.submat(offset, 0, offset + cell2GateForgetWeight.n_elem - 1, 0) =
        sum(forgetGateError % cell.slice(this->PreviousStep()), 1);
    gradient.submat(offset + cell2GateForgetWeight.n_elem, 0, offset +
        cell2GateForgetWeight.n_elem + cell2GateInputWeight.n_elem - 1, 0) =
        sum(inputGateError % cell.slice(this->PreviousStep()), 1);
  }
  else
  {
    gradient.submat(offset, 0, offset +
        cell2GateForgetWeight.n_elem - 1, 0).zeros();
    gradient.submat(offset + cell2GateForgetWeight.n_elem, 0, offset +
        cell2GateForgetWeight.n_elem +
        cell2GateInputWeight.n_elem - 1, 0).zeros();
  }
}

template<typename MatType>
template<typename Archive>
void LSTMType<MatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<RecurrentLayer<MatType>>(this));

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));

  // Clear recurrent state if we are loading.
  if (Archive::is_loading::value)
  {
    inputGateActivation.clear();
    forgetGateActivation.clear();
    outputGateActivation.clear();
    hiddenLayerActivation.clear();
    cellActivation.clear();
    forgetGateError.clear();
    outputGateError.clear();
    outParameter.clear();
    inputCellError.clear();
    inputGateError.clear();
    hiddenError.clear();
  }
}

} // namespace mlpack

#endif
