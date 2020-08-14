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
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
LSTM<InputDataType, OutputDataType>::LSTM()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
LSTM<InputDataType, OutputDataType>::LSTM(
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
  weights.set_size(4 * outSize * inSize + 7 * outSize +
      4 * outSize * outSize, 1);
}

template<typename InputDataType, typename OutputDataType>
void LSTM<InputDataType, OutputDataType>::ResetCell(const size_t size)
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
  if (inputGate.is_empty() || inputGate.n_cols < rhoBatchSize)
  {
    inputGate.set_size(outSize, rhoBatchSize);
    forgetGate.set_size(outSize, rhoBatchSize);
    hiddenLayer.set_size(outSize, rhoBatchSize);
    outputGate.set_size(outSize, rhoBatchSize);

    inputGateActivation.set_size(outSize, rhoBatchSize);
    forgetGateActivation.set_size(outSize, rhoBatchSize);
    outputGateActivation.set_size(outSize, rhoBatchSize);
    hiddenLayerActivation.set_size(outSize, rhoBatchSize);

    cellActivation.set_size(outSize, rhoBatchSize);
    prevError.set_size(4 * outSize, batchSize);

    if (cell.is_empty())
    {
      cell = arma::zeros(outSize, size * batchSize);
      outParameter = arma::zeros<OutputDataType>(
          outSize, (size + 1) * batchSize);
    }
    else
    {
      // To preserve the leading zeros, recreate the object according to given
      // size specifications, while preserving the elements as well as the
      // layout of the elements.
      cell.resize(outSize, size * batchSize);
      outParameter.resize(outSize, (size + 1) * batchSize);
    }
  }
}

template<typename InputDataType, typename OutputDataType>
void LSTM<InputDataType, OutputDataType>::Reset()
{
  // Set the weight parameter for the output gate.
  input2GateOutputWeight = OutputDataType(weights.memptr(), outSize, inSize,
      false, false);
  input2GateOutputBias = OutputDataType(weights.memptr() +
      input2GateOutputWeight.n_elem, outSize, 1, false, false);
  size_t offset = input2GateOutputWeight.n_elem + input2GateOutputBias.n_elem;

  // Set the weight parameter for the forget gate.
  input2GateForgetWeight = OutputDataType(weights.memptr() + offset,
      outSize, inSize, false, false);
  input2GateForgetBias = OutputDataType(weights.memptr() +
      offset + input2GateForgetWeight.n_elem, outSize, 1, false, false);
  offset += input2GateForgetWeight.n_elem + input2GateForgetBias.n_elem;

  // Set the weight parameter for the input gate.
  input2GateInputWeight = OutputDataType(weights.memptr() +
      offset, outSize, inSize, false, false);
  input2GateInputBias = OutputDataType(weights.memptr() +
      offset + input2GateInputWeight.n_elem, outSize, 1, false, false);
  offset += input2GateInputWeight.n_elem + input2GateInputBias.n_elem;

  // Set the weight parameter for the hidden gate.
  input2HiddenWeight = OutputDataType(weights.memptr() +
      offset, outSize, inSize, false, false);
  input2HiddenBias = OutputDataType(weights.memptr() +
      offset + input2HiddenWeight.n_elem, outSize, 1, false, false);
  offset += input2HiddenWeight.n_elem + input2HiddenBias.n_elem;

  // Set the weight parameter for the output multiplication.
  output2GateOutputWeight = OutputDataType(weights.memptr() +
      offset, outSize, outSize, false, false);
  offset += output2GateOutputWeight.n_elem;

  // Set the weight parameter for the output multiplication.
  output2GateForgetWeight = OutputDataType(weights.memptr() +
      offset, outSize, outSize, false, false);
  offset += output2GateForgetWeight.n_elem;

  // Set the weight parameter for the input multiplication.
  output2GateInputWeight = OutputDataType(weights.memptr() +
      offset, outSize, outSize, false, false);
  offset += output2GateInputWeight.n_elem;

  // Set the weight parameter for the hidden multiplication.
  output2HiddenWeight = OutputDataType(weights.memptr() +
      offset, outSize, outSize, false, false);
  offset += output2HiddenWeight.n_elem;

  // Set the weight parameter for the cell multiplication.
  cell2GateOutputWeight = OutputDataType(weights.memptr() +
      offset, outSize, 1, false, false);
  offset += cell2GateOutputWeight.n_elem;

  // Set the weight parameter for the cell - forget gate multiplication.
  cell2GateForgetWeight = OutputDataType(weights.memptr() +
      offset, outSize, 1, false, false);
  offset += cell2GateOutputWeight.n_elem;

  // Set the weight parameter for the cell - input gate multiplication.
  cell2GateInputWeight = OutputDataType(weights.memptr() +
      offset, outSize, 1, false, false);
}

// Forward when cellState is not needed.
template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void LSTM<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  //! Locally-stored cellState.
  OutputType cellState;
  Forward(input, output, cellState, false);
}

// Forward when cellState is needed overloaded LSTM::Forward().
template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void LSTM<InputDataType, OutputDataType>::Forward(const InputType& input,
                                                  OutputType& output,
                                                  OutputType& cellState,
                                                  bool useCellState)
{
  // Check if the batch size changed, the number of cols is defines the input
  // batch size.
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    batchStep = batchSize - 1;
    ResetCell(rhoSize);
  }

  inputGate.cols(forwardStep, forwardStep + batchStep) = input2GateInputWeight *
      input + output2GateInputWeight * outParameter.cols(forwardStep,
      forwardStep + batchStep);
  inputGate.cols(forwardStep, forwardStep + batchStep).each_col() +=
      input2GateInputBias;

  forgetGate.cols(forwardStep, forwardStep + batchStep) = input2GateForgetWeight
      * input + output2GateForgetWeight * outParameter.cols(
      forwardStep, forwardStep + batchStep);
  forgetGate.cols(forwardStep, forwardStep + batchStep).each_col() +=
      input2GateForgetBias;

  if (forwardStep > 0)
  {
    if (useCellState)
    {
      if (!cellState.is_empty())
      {
        cell.cols(forwardStep - batchSize,
            forwardStep - batchSize + batchStep) = cellState;
      }
      else
      {
        throw std::runtime_error("Cell parameter is empty.");
      }
    }
    inputGate.cols(forwardStep, forwardStep + batchStep) +=
        arma::repmat(cell2GateInputWeight, 1, batchSize) %
        cell.cols(forwardStep - batchSize, forwardStep - batchSize + batchStep);

    forgetGate.cols(forwardStep, forwardStep + batchStep) +=
        arma::repmat(cell2GateForgetWeight, 1, batchSize) %
        cell.cols(forwardStep - batchSize, forwardStep - batchSize + batchStep);
  }

  inputGateActivation.cols(forwardStep, forwardStep + batchStep) = 1.0 /
      (1 + arma::exp(-inputGate.cols(forwardStep, forwardStep + batchStep)));

  forgetGateActivation.cols(forwardStep, forwardStep + batchStep) = 1.0 /
      (1 + arma::exp(-forgetGate.cols(forwardStep, forwardStep + batchStep)));

  hiddenLayer.cols(forwardStep, forwardStep + batchStep) = input2HiddenWeight *
      input + output2HiddenWeight * outParameter.cols(
      forwardStep, forwardStep + batchStep);

  hiddenLayer.cols(forwardStep, forwardStep + batchStep).each_col() +=
      input2HiddenBias;

  hiddenLayerActivation.cols(forwardStep, forwardStep + batchStep) =
      arma::tanh(hiddenLayer.cols(forwardStep, forwardStep + batchStep));

  if (forwardStep == 0)
  {
    cell.cols(forwardStep, forwardStep + batchStep) =
        inputGateActivation.cols(forwardStep, forwardStep + batchStep) %
        hiddenLayerActivation.cols(forwardStep, forwardStep + batchStep);
  }
  else
  {
    cell.cols(forwardStep, forwardStep + batchStep) =
        forgetGateActivation.cols(forwardStep, forwardStep + batchStep) %
        cell.cols(forwardStep - batchSize, forwardStep - batchSize + batchStep)
        + inputGateActivation.cols(forwardStep, forwardStep + batchStep) %
        hiddenLayerActivation.cols(forwardStep, forwardStep + batchStep);
  }

  outputGate.cols(forwardStep, forwardStep + batchStep) = input2GateOutputWeight
      * input + output2GateOutputWeight * outParameter.cols(
      forwardStep, forwardStep + batchStep) + cell.cols(forwardStep,
      forwardStep + batchStep).each_col() % cell2GateOutputWeight;

  outputGate.cols(forwardStep, forwardStep + batchStep).each_col() +=
      input2GateOutputBias;

  outputGateActivation.cols(forwardStep, forwardStep + batchStep) = 1.0 /
      (1 + arma::exp(-outputGate.cols(forwardStep, forwardStep + batchStep)));

  cellActivation.cols(forwardStep, forwardStep + batchStep) =
      arma::tanh(cell.cols(forwardStep, forwardStep + batchStep));

  outParameter.cols(forwardStep + batchSize,
      forwardStep + batchSize + batchStep) =
      cellActivation.cols(forwardStep, forwardStep + batchStep) %
      outputGateActivation.cols(forwardStep, forwardStep + batchStep);

  output = OutputType(outParameter.memptr() +
      (forwardStep + batchSize) * outSize, outSize, batchSize, false, false);

  cellState = OutputType(cell.memptr() +
      forwardStep * outSize, outSize, batchSize, false, false);

  forwardStep += batchSize;
  if ((forwardStep / batchSize) == bpttSteps)
  {
    forwardStep = 0;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename ErrorType, typename GradientType>
void LSTM<InputDataType, OutputDataType>::Backward(
  const InputType& /* input */, const ErrorType& gy, GradientType& g)
{
  ErrorType gyLocal;
  if (gradientStepIdx > 0)
  {
    gyLocal = gy + prevError;
  }
  else
  {
    // Make an alias.
    gyLocal = ErrorType(((ErrorType&) gy).memptr(), gy.n_rows, gy.n_cols, false,
        false);
  }

  outputGateError =
      gyLocal % cellActivation.cols(backwardStep - batchStep, backwardStep) %
      (outputGateActivation.cols(backwardStep - batchStep, backwardStep) %
      (1.0 - outputGateActivation.cols(backwardStep - batchStep,
      backwardStep)));

  OutputDataType cellError = gyLocal %
      outputGateActivation.cols(backwardStep - batchStep, backwardStep) %
      (1 - arma::pow(cellActivation.cols(backwardStep -
      batchStep, backwardStep), 2)) + outputGateError.each_col() %
      cell2GateOutputWeight;

  if (gradientStepIdx > 0)
  {
    cellError += inputCellError;
  }

  if (backwardStep > batchStep)
  {
    forgetGateError = cell.cols((backwardStep - batchSize) - batchStep,
      (backwardStep - batchSize)) % cellError % (forgetGateActivation.cols(
      backwardStep - batchStep, backwardStep) % (1.0 -
      forgetGateActivation.cols(backwardStep - batchStep, backwardStep)));
  }
  else
  {
    forgetGateError.zeros();
  }

  inputGateError = hiddenLayerActivation.cols(backwardStep - batchStep,
      backwardStep) % cellError %
      (inputGateActivation.cols(backwardStep - batchStep, backwardStep) %
      (1.0 - inputGateActivation.cols(backwardStep - batchStep, backwardStep)));

  hiddenError = inputGateActivation.cols(backwardStep - batchStep,
      backwardStep) % cellError % (1 - arma::pow(hiddenLayerActivation.cols(
      backwardStep - batchStep, backwardStep), 2));

  inputCellError = forgetGateActivation.cols(backwardStep - batchStep,
      backwardStep) % cellError + forgetGateError.each_col() %
      cell2GateForgetWeight + inputGateError.each_col() % cell2GateInputWeight;

  g = input2GateInputWeight.t() * inputGateError +
      input2HiddenWeight.t() * hiddenError +
      input2GateForgetWeight.t() * forgetGateError +
      input2GateOutputWeight.t() * outputGateError;

  prevError = output2GateOutputWeight.t() * outputGateError +
      output2GateForgetWeight.t() * forgetGateError +
      output2GateInputWeight.t() * inputGateError +
      output2HiddenWeight.t() * hiddenError;

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
void LSTM<InputDataType, OutputDataType>::Gradient(
    const InputType& input,
    const ErrorType& /* error */,
    GradientType& gradient)
{
  // Input2GateOutputWeight and input2GateOutputBias gradients.
  gradient.submat(0, 0, input2GateOutputWeight.n_elem - 1, 0) =
      arma::vectorise(outputGateError * input.t());
  gradient.submat(input2GateOutputWeight.n_elem, 0,
      input2GateOutputWeight.n_elem + input2GateOutputBias.n_elem - 1, 0) =
      arma::sum(outputGateError, 1);
  size_t offset = input2GateOutputWeight.n_elem + input2GateOutputBias.n_elem;

  // Input2GateForgetWeight and input2GateForgetBias gradients.
  gradient.submat(offset, 0, offset + input2GateForgetWeight.n_elem - 1, 0) =
      arma::vectorise(forgetGateError * input.t());
  gradient.submat(offset + input2GateForgetWeight.n_elem, 0,
      offset + input2GateForgetWeight.n_elem +
      input2GateForgetBias.n_elem - 1, 0) = arma::sum(forgetGateError, 1);
  offset += input2GateForgetWeight.n_elem + input2GateForgetBias.n_elem;

  // Input2GateInputWeight and input2GateInputBias gradients.
  gradient.submat(offset, 0, offset + input2GateInputWeight.n_elem - 1, 0) =
      arma::vectorise(inputGateError * input.t());
  gradient.submat(offset + input2GateInputWeight.n_elem, 0,
      offset + input2GateInputWeight.n_elem +
      input2GateInputBias.n_elem - 1, 0) = arma::sum(inputGateError, 1);
  offset += input2GateInputWeight.n_elem + input2GateInputBias.n_elem;

  // Input2HiddenWeight and input2HiddenBias gradients.
  gradient.submat(offset, 0, offset + input2HiddenWeight.n_elem - 1, 0) =
      arma::vectorise(hiddenError * input.t());
  gradient.submat(offset + input2HiddenWeight.n_elem, 0,
      offset + input2HiddenWeight.n_elem + input2HiddenBias.n_elem - 1, 0) =
      arma::sum(hiddenError, 1);
  offset += input2HiddenWeight.n_elem + input2HiddenBias.n_elem;

  // Output2GateOutputWeight gradients.
  gradient.submat(offset, 0, offset + output2GateOutputWeight.n_elem - 1, 0) =
      arma::vectorise(outputGateError *
      outParameter.cols(gradientStep - batchStep, gradientStep).t());
  offset += output2GateOutputWeight.n_elem;

  // Output2GateForgetWeight gradients.
  gradient.submat(offset, 0, offset + output2GateForgetWeight.n_elem - 1, 0) =
      arma::vectorise(forgetGateError *
      outParameter.cols(gradientStep - batchStep, gradientStep).t());
  offset += output2GateForgetWeight.n_elem;

  // Output2GateInputWeight gradients.
  gradient.submat(offset, 0, offset + output2GateInputWeight.n_elem - 1, 0) =
      arma::vectorise(inputGateError *
      outParameter.cols(gradientStep - batchStep, gradientStep).t());
  offset += output2GateInputWeight.n_elem;

  // Output2HiddenWeight gradients.
  gradient.submat(offset, 0, offset + output2HiddenWeight.n_elem - 1, 0) =
      arma::vectorise(hiddenError *
      outParameter.cols(gradientStep - batchStep, gradientStep).t());
  offset += output2HiddenWeight.n_elem;

  // Cell2GateOutputWeight gradients.
  gradient.submat(offset, 0, offset + cell2GateOutputWeight.n_elem - 1, 0) =
      arma::sum(outputGateError %
      cell.cols(gradientStep - batchStep, gradientStep), 1);
  offset += cell2GateOutputWeight.n_elem;

  // Cell2GateForgetWeight and cell2GateInputWeight gradients.
  if (gradientStep > batchStep)
  {
    gradient.submat(offset, 0, offset + cell2GateForgetWeight.n_elem - 1, 0) =
        arma::sum(forgetGateError %
                  cell.cols((gradientStep - batchSize) - batchStep,
                            (gradientStep - batchSize)), 1);
    gradient.submat(offset + cell2GateForgetWeight.n_elem, 0, offset +
        cell2GateForgetWeight.n_elem + cell2GateInputWeight.n_elem - 1, 0) =
        arma::sum(inputGateError %
                  cell.cols((gradientStep - batchSize) - batchStep,
                            (gradientStep - batchSize)), 1);
  }
  else
  {
    gradient.submat(offset, 0, offset +
        cell2GateForgetWeight.n_elem - 1, 0).zeros();
    gradient.submat(offset + cell2GateForgetWeight.n_elem, 0, offset +
        cell2GateForgetWeight.n_elem +
        cell2GateInputWeight.n_elem - 1, 0).zeros();
  }

  if (gradientStep == 0)
  {
    gradientStep = batchSize * bpttSteps - 1;
  }
  else
  {
    gradientStep -= batchSize;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void LSTM<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(weights);
  ar & CEREAL_NVP(inSize);
  ar & CEREAL_NVP(outSize);
  ar & CEREAL_NVP(rho);
  ar & CEREAL_NVP(bpttSteps);
  ar & CEREAL_NVP(batchSize);
  ar & CEREAL_NVP(batchStep);
  ar & CEREAL_NVP(forwardStep);
  ar & CEREAL_NVP(backwardStep);
  ar & CEREAL_NVP(gradientStep);
  ar & CEREAL_NVP(gradientStepIdx);
  ar & CEREAL_NVP(cell);
  ar & CEREAL_NVP(inputGateActivation);
  ar & CEREAL_NVP(forgetGateActivation);
  ar & CEREAL_NVP(outputGateActivation);
  ar & CEREAL_NVP(hiddenLayerActivation);
  ar & CEREAL_NVP(cellActivation);
  ar & CEREAL_NVP(prevError);
  ar & CEREAL_NVP(outParameter);
}

} // namespace ann
} // namespace mlpack

#endif
