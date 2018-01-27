/**
 * @file fast_lstm_impl.hpp
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
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
FastLSTM<InputDataType, OutputDataType>::FastLSTM()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
FastLSTM<InputDataType, OutputDataType>::FastLSTM(
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
  weights.set_size(
      4 * outSize * inSize + 4 * outSize + 4 * outSize * outSize, 1);
}

template<typename InputDataType, typename OutputDataType>
void FastLSTM<InputDataType, OutputDataType>::Reset()
{
  // Set the weight parameter for the input to gate layer (linear layer) using
  // the overall layer parameter matrix.
  input2GateWeight = OutputDataType(weights.memptr(),
      4 * outSize, inSize, false, false);
  input2GateBias = OutputDataType(weights.memptr() + input2GateWeight.n_elem,
      4 * outSize, 1, false, false);

  // Set the weight parameter for the output to gate layer
  // (linear no bias layer) using the overall layer parameter matrix.
  output2GateWeight = OutputDataType(weights.memptr() + input2GateWeight.n_elem
      + input2GateBias.n_elem, 4 * outSize, outSize, false, false);
}

template<typename InputDataType, typename OutputDataType>
void FastLSTM<InputDataType, OutputDataType>::ResetCell(const size_t size)
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
  if (gate.is_empty() || gate.n_cols != rhoBatchSize)
  {
    gate.set_size(4 * outSize, rhoBatchSize);
    gateActivation.set_size(outSize * 3, rhoBatchSize);
    stateActivation.set_size(outSize, rhoBatchSize);
    cellActivation.set_size(outSize, rhoBatchSize);
    prevError.set_size(4 * outSize, batchSize);

    if (prevOutput.is_empty())
    {
      prevOutput = arma::zeros<OutputDataType>(outSize, batchSize);
      cell = arma::zeros(outSize, size * batchSize);
      cellActivationError = arma::zeros<OutputDataType>(outSize, batchSize);
      outParameter = arma::zeros<OutputDataType>(
          outSize, (size + 1) * batchSize);
    }
    else
    {
      // To preserve the leading zeros, recreate the object according to given
      // size specifications, while preserving the elements as well as the
      // layout of the elements.
      prevOutput.resize(outSize, batchSize);
      cell.resize(outSize, size * batchSize);
      cellActivationError.resize(outSize, batchSize);
      outParameter.resize(outSize, (size + 1) * batchSize);
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void FastLSTM<InputDataType, OutputDataType>::Forward(
    InputType&& input, OutputType&& output)
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

  FastSigmoid(std::move(
      gate.submat(0, forwardStep, 3 * outSize - 1, forwardStep + batchStep)),
      std::move(gateActivation.cols(forwardStep, forwardStep + batchStep)));

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

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename ErrorType, typename GradientType>
void FastLSTM<InputDataType, OutputDataType>::Backward(
  const InputType&& /* input */, ErrorType&& gy, GradientType&& g)
{
  if (gradientStepIdx > 0)
  {
    gy += output2GateWeight.t() * prevError;
  }

  cellActivationError = gy % gateActivation.submat(outSize,
      backwardStep - batchStep, 2 * outSize - 1, backwardStep) %
      (1 - arma::pow(cellActivation.cols(backwardStep - batchStep,
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
      outSize - 1, backwardStep) % cellActivationError % (1 - arma::pow(
      stateActivation.cols(backwardStep - batchStep, backwardStep), 2));

  prevError.submat(outSize, 0, 2 * outSize - 1, batchStep) =
      cellActivation.cols(backwardStep - batchStep,
      backwardStep) % gy % gateActivation.submat(
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

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename ErrorType, typename GradientType>
void FastLSTM<InputDataType, OutputDataType>::Gradient(
    InputType&& input, ErrorType&& /* error */, GradientType&& gradient)
{
  // Gradient of the input to gate layer.
  gradient.submat(0, 0, input2GateWeight.n_elem - 1, 0) =
      arma::vectorise(prevError * input.t());

  gradient.submat(input2GateWeight.n_elem, 0, input2GateWeight.n_elem +
      input2GateBias.n_elem - 1, 0) = arma::sum(prevError, 1);

  // Gradient of the output to gate layer.
  gradient.submat(input2GateWeight.n_elem + input2GateBias.n_elem, 0,
      gradient.n_elem - 1, 0) = arma::vectorise(prevError *
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

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void FastLSTM<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(weights);
  ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);
  ar & BOOST_SERIALIZATION_NVP(rho);
  ar & BOOST_SERIALIZATION_NVP(bpttSteps);
  ar & BOOST_SERIALIZATION_NVP(batchSize);
  ar & BOOST_SERIALIZATION_NVP(batchStep);
  ar & BOOST_SERIALIZATION_NVP(forwardStep);
  ar & BOOST_SERIALIZATION_NVP(backwardStep);
  ar & BOOST_SERIALIZATION_NVP(gradientStep);
  ar & BOOST_SERIALIZATION_NVP(gradientStepIdx);
  ar & BOOST_SERIALIZATION_NVP(cell);
  ar & BOOST_SERIALIZATION_NVP(stateActivation);
  ar & BOOST_SERIALIZATION_NVP(gateActivation);
  ar & BOOST_SERIALIZATION_NVP(gate);
  ar & BOOST_SERIALIZATION_NVP(cellActivation);
  ar & BOOST_SERIALIZATION_NVP(forgetGateError);
  ar & BOOST_SERIALIZATION_NVP(prevError);
  ar & BOOST_SERIALIZATION_NVP(outParameter);
}

} // namespace ann
} // namespace mlpack

#endif
