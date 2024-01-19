/**
 * @file methods/ann/layer/fast_lstm_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Fast LSTM class, which implements a fast lstm network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_FAST_LSTM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_FAST_LSTM_IMPL_HPP

// In case it hasn't yet been included.
#include "fast_lstm.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
FastLSTMType<InputType, OutputType>::FastLSTMType()
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
FastLSTMType<InputType, OutputType>::FastLSTMType(
    const size_t inSize, const size_t outSize, const size_t rho) :
    inSize(inSize),
    outSize(outSize),
    rho(rho),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    batchSize(0),
    batchStep(0),
    gradientStepIdx(0),
    rhoSize(rho),
    bpttSteps(0)
{
  // Weights for: input to gate layer (4 * outsize * inSize + 4 * outsize)
  // and output to gate (4 * outSize).
  weights.set_size(WeightSize(), 1);
}

template<typename InputType, typename OutputType>
FastLSTMType<InputType, OutputType>::FastLSTMType(const FastLSTMType& layer) :
    inSize(layer.inSize),
    outSize(layer.outSize),
    rho(layer.rho),
    forwardStep(layer.forwardStep),
    backwardStep(layer.backwardStep),
    gradientStep(layer.gradientStep),
    weights(layer.weights),
    batchSize(layer.batchSize),
    batchStep(layer.batchStep),
    gradientStepIdx(layer.gradientStepIdx),
    rhoSize(layer.rho),
    bpttSteps(layer.bpttSteps)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
FastLSTMType<InputType, OutputType>::FastLSTMType(FastLSTMType&& layer) :
    inSize(std::move(layer.inSize)),
    outSize(std::move(layer.outSize)),
    rho(std::move(layer.rho)),
    forwardStep(std::move(layer.forwardStep)),
    backwardStep(std::move(layer.backwardStep)),
    gradientStep(std::move(layer.gradientStep)),
    weights(std::move(layer.weights)),
    batchSize(std::move(layer.batchSize)),
    batchStep(std::move(layer.batchStep)),
    gradientStepIdx(std::move(layer.gradientStepIdx)),
    rhoSize(std::move(layer.rho)),
    bpttSteps(std::move(layer.bpttSteps))
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
FastLSTMType<InputType, OutputType>&
FastLSTMType<InputType, OutputType>::operator=(const FastLSTMType& layer)
{
  if (this != &layer)
  {
    inSize = layer.inSize;
    outSize = layer.outSize;
    rho = layer.rho;
    forwardStep = layer.forwardStep;
    backwardStep = layer.backwardStep;
    gradientStep = layer.gradientStep;
    weights = layer.weights;
    batchSize = layer.batchSize;
    batchStep = layer.batchStep;
    gradientStepIdx = layer.gradientStepIdx;
    rhoSize = layer.rho;
    bpttSteps = layer.bpttSteps;
  }
  return *this;
}

template<typename InputType, typename OutputType>
FastLSTMType<InputType, OutputType>&
FastLSTMType<InputType, OutputType>::operator=(FastLSTMType&& layer)
{
  if (this != &layer)
  {
    inSize = std::move(layer.inSize);
    outSize = std::move(layer.outSize);
    rho = std::move(layer.rho);
    forwardStep = std::move(layer.forwardStep);
    backwardStep = std::move(layer.backwardStep);
    gradientStep = std::move(layer.gradientStep);
    weights = std::move(layer.weights);
    batchSize = std::move(layer.batchSize);
    batchStep = std::move(layer.batchStep);
    gradientStepIdx = std::move(layer.gradientStepIdx);
    rhoSize = std::move(layer.rho);
    bpttSteps = std::move(layer.bpttSteps);
  }
  return *this;
}

template<typename InputType, typename OutputType>
void FastLSTMType<InputType, OutputType>::Reset()
{
  // Set the weight parameter for the input to gate layer (linear layer) using
  // the overall layer parameter matrix.
  input2GateWeight = OutputType(weights.memptr(),
      4 * outSize, inSize, false, false);
  input2GateBias = OutputType(weights.memptr() + input2GateWeight.n_elem,
      4 * outSize, 1, false, false);

  // Set the weight parameter for the output to gate layer
  // (linear no bias layer) using the overall layer parameter matrix.
  output2GateWeight = OutputType(weights.memptr() + input2GateWeight.n_elem
      + input2GateBias.n_elem, 4 * outSize, outSize, false, false);
}

template<typename InputType, typename OutputType>
void FastLSTMType<InputType, OutputType>::ResetCell(const size_t size)
{
  if (size == std::numeric_limits<size_t>::max())
    return;

  rhoSize = size;

  if (batchSize == 0)
    return;

  bpttSteps = std::min(rho, rhoSize);
  forwardStep = 0;
  gradientStepIdx = 0;
  backwardStep = batchSize * size - 1;
  gradientStep = batchSize * size - 1;

  const size_t rhoBatchSize = size * batchSize;

  // Make sure all of the matrices we use to store state are at least as large
  // as we need.
  gate.set_size(4 * outSize, rhoBatchSize);
  gateActivation.set_size(outSize * 3, rhoBatchSize);
  stateActivation.set_size(outSize, rhoBatchSize);
  cellActivation.set_size(outSize, rhoBatchSize);
  prevError.set_size(4 * outSize, batchSize);

  // Reset stored state to zeros.
  prevOutput.zeros(outSize, batchSize);
  cell.zeros(outSize, size * batchSize);
  cellActivationError.zeros(outSize, batchSize);
  outParameter.zeros(outSize, (size + 1) * batchSize);
}

template<typename InputType, typename OutputType>
void FastLSTMType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  // Check if the batch size changed, the number of cols is defines the input
  // batch size.
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    batchStep = batchSize - 1;
    ResetCell(rhoSize);
  }

  gate.cols(forwardStep, forwardStep + batchStep) = input2GateWeight * input +
      output2GateWeight * outParameter.cols(
      forwardStep, forwardStep + batchStep);
  gate.cols(forwardStep, forwardStep + batchStep).each_col() += input2GateBias;

  InputType sigmoidOut(gateActivation.colptr(forwardStep),
      gateActivation.n_rows, batchStep, false, false);
  FastSigmoid(
      gate.submat(0, forwardStep, 3 * outSize - 1, forwardStep + batchStep),
      sigmoidOut);

  stateActivation.cols(forwardStep, forwardStep + batchStep) = arma::tanh(
      gate.submat(3 * outSize, forwardStep, 4 * outSize - 1,
      forwardStep + batchStep));

  // Update the cell: cmul1 + cmul2
  // where cmul1 is input gate * hidden state and
  // cmul2 is forget gate * cell (prevCell).
  if (forwardStep == 0)
  {
    cell.cols(forwardStep, forwardStep + batchStep) =
        gateActivation.submat(0, forwardStep, outSize - 1,
        forwardStep + batchStep) %
        stateActivation.cols(forwardStep, forwardStep + batchStep);
  }
  else
  {
    cell.cols(forwardStep, forwardStep + batchStep) =
        gateActivation.submat(0, forwardStep, outSize - 1,
        forwardStep + batchStep) %
        stateActivation.cols(forwardStep, forwardStep + batchStep) +
        gateActivation.submat(2 * outSize, forwardStep, 3 * outSize - 1,
        forwardStep + batchStep) %
        cell.cols(forwardStep - batchSize, forwardStep - batchSize + batchStep);
  }

  cellActivation.cols(forwardStep, forwardStep + batchStep) =
      arma::tanh(cell.cols(forwardStep, forwardStep + batchStep));

  outParameter.cols(forwardStep + batchSize,
      forwardStep + batchSize + batchStep) = cellActivation.cols(
      forwardStep, forwardStep + batchStep) % gateActivation.submat(
      outSize, forwardStep, 2 * outSize - 1, forwardStep + batchStep);

  output = OutputType(outParameter.memptr() +
      (forwardStep + batchSize) * outSize, outSize, batchSize, false, false);

  forwardStep += batchSize;
  if ((forwardStep / batchSize) == bpttSteps)
  {
    forwardStep = 0;
  }
}

template<typename InputType, typename OutputType>
void FastLSTMType<InputType, OutputType>::Backward(
  const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  OutputType gyLocal;
  if (gradientStepIdx > 0)
  {
    gyLocal = gy + output2GateWeight.t() * prevError;
  }
  else
  {
    gyLocal = OutputType(((OutputType&) gy).memptr(), gy.n_rows, gy.n_cols,
        false, false);
  }

  cellActivationError = gyLocal % gateActivation.submat(outSize,
      backwardStep - batchStep, 2 * outSize - 1, backwardStep) %
      (1 - pow(cellActivation.cols(backwardStep - batchStep,
      backwardStep), 2));

  if (gradientStepIdx > 0)
    cellActivationError += forgetGateError;

  forgetGateError = gateActivation.submat(2 * outSize,
      backwardStep - batchStep, 3 * outSize - 1, backwardStep) %
      cellActivationError;

  if (backwardStep > batchStep)
  {
    prevError.submat(2 * outSize, 0, 3 * outSize - 1, batchStep) =
        cell.cols((backwardStep - batchSize) - batchStep,
        (backwardStep - batchSize)) % cellActivationError %
        gateActivation.submat(2 * outSize, backwardStep - batchStep,
        3 * outSize - 1, backwardStep) % (1.0 - gateActivation.submat(
        2 * outSize, backwardStep - batchStep, 3 * outSize - 1, backwardStep));
  }
  else
  {
    prevError.submat(2 * outSize, 0, 3 * outSize - 1, batchStep).zeros();
  }

  prevError.submat(0, 0, outSize - 1, batchStep) =
      stateActivation.cols(backwardStep - batchStep,
      backwardStep) % cellActivationError % gateActivation.submat(
      0, backwardStep - batchStep, outSize - 1, backwardStep) %
      (1.0 - gateActivation.submat(
      0, backwardStep - batchStep, outSize - 1, backwardStep));

  prevError.submat(3 * outSize, 0, 4 * outSize - 1, batchStep) =
      gateActivation.submat(0, backwardStep - batchStep,
      outSize - 1, backwardStep) % cellActivationError % (1 - pow(
      stateActivation.cols(backwardStep - batchStep, backwardStep), 2));

  prevError.submat(outSize, 0, 2 * outSize - 1, batchStep) =
      cellActivation.cols(backwardStep - batchStep,
      backwardStep) % gyLocal % gateActivation.submat(
       outSize, backwardStep - batchStep, 2 * outSize - 1, backwardStep) %
      (1.0 - gateActivation.submat(
      outSize, backwardStep - batchStep, 2 * outSize - 1, backwardStep));

  g = input2GateWeight.t() * prevError;

  backwardStep -= batchSize;
  gradientStepIdx++;
  if (gradientStepIdx == bpttSteps)
  {
    backwardStep = bpttSteps - 1;
    gradientStepIdx = 0;
  }
}

template<typename InputType, typename OutputType>
void FastLSTMType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& /* error */,
    OutputType& gradient)
{
  // Gradient of the input to gate layer.
  gradient.submat(0, 0, input2GateWeight.n_elem - 1, 0) =
      vectorise(prevError * input.t());

  gradient.submat(input2GateWeight.n_elem, 0, input2GateWeight.n_elem +
      input2GateBias.n_elem - 1, 0) = sum(prevError, 1);

  // Gradient of the output to gate layer.
  gradient.submat(input2GateWeight.n_elem + input2GateBias.n_elem, 0,
      gradient.n_elem - 1, 0) = vectorise(prevError *
      outParameter.cols(gradientStep - batchStep, gradientStep).t());

  if (gradientStep > batchStep)
  {
    gradientStep -= batchSize;
  }
  else
  {
    gradientStep = batchSize * bpttSteps - 1;
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void FastLSTMType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(weights));
  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(rho));
  ar(CEREAL_NVP(bpttSteps));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(batchStep));
  ar(CEREAL_NVP(forwardStep));
  ar(CEREAL_NVP(backwardStep));
  ar(CEREAL_NVP(gradientStep));
  ar(CEREAL_NVP(gradientStepIdx));
  ar(CEREAL_NVP(cell));
  ar(CEREAL_NVP(stateActivation));
  ar(CEREAL_NVP(gateActivation));
  ar(CEREAL_NVP(gate));
  ar(CEREAL_NVP(cellActivation));
  ar(CEREAL_NVP(forgetGateError));
  ar(CEREAL_NVP(prevError));

  // Restore aliases.
  if (Archive::is_loading::value)
    Reset();
}

} // namespace mlpack

#endif
