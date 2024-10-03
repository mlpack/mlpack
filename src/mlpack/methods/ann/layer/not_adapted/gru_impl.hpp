/**
 * @file methods/ann/layer/gru_impl.hpp
 * @author Sumedh Ghaisas
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

template<typename InputType, typename OutputType>
GRU<InputType, OutputType>::GRU()
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType>
GRU<InputType, OutputType>::GRU(
    const size_t inSize,
    const size_t outSize,
    const size_t rho) :
    inSize(inSize),
    outSize(outSize),
    rho(rho),
    batchSize(1),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0)
{
  // Input specific linear layers(for zt, rt, ot).
  input2GateModule = new LinearType<InputType, OutputType>(inSize, 3 * outSize);

  // Previous output gates (for zt and rt).
  output2GateModule = new LinearNoBiasType<InputType, OutputType>(outSize,
      2 * outSize);

  // Previous output gate for ot.
  outputHidden2GateModule = new LinearNoBiasType<InputType, OutputType>(outSize,
      outSize);

  network.push_back(input2GateModule);
  network.push_back(output2GateModule);
  network.push_back(outputHidden2GateModule);

  inputGateModule = new SigmoidLayer<InputType, OutputType>();
  forgetGateModule = new SigmoidLayer<InputType, OutputType>();
  hiddenStateModule = new TanHLayer<InputType, OutputType>();

  network.push_back(inputGateModule);
  network.push_back(hiddenStateModule);
  network.push_back(forgetGateModule);

  prevError = zeros<OutputType>(3 * outSize, batchSize);

  allZeros = zeros<OutputType>(outSize, batchSize);

  outParameter.emplace_back(allZeros.memptr(),
      allZeros.n_rows, allZeros.n_cols, false, true);

  prevOutput = outParameter.begin();
  backIterator = outParameter.end();
  gradIterator = outParameter.end();
}

template<typename InputType, typename OutputType>
void GRU<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    prevError.resize(3 * outSize, batchSize);
    allZeros.zeros(outSize, batchSize);
    // Batch size better not change during an iteration...
    if (outParameter.size() > 1)
    {
      Log::Fatal << "GRU<>::Forward(): batch size cannot change during a "
          << "forward pass!" << std::endl;
    }

    outParameter.clear();
    outParameter.emplace_back(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true);

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  // Process the input linearly(zt, rt, ot).
  input2GateModule->Forward(input, input2GateModule->OutputParameter());

  // Process the output(zt, rt) linearly.
  output2GateModule->Forward(*prevOutput, output2GateModule->OutputParameter());

  // Merge the outputs(zt and rt).
  output = input2GateModule->OutputParameter().submat(0, 0, 2 * outSize - 1,
      batchSize - 1) + output2GateModule->OutputParameter();

  // Pass the first outSize through inputGate(it).
  inputGateModule->Forward(output.submat(0, 0, 1 * outSize - 1, batchSize - 1),
      inputGateModule->OutputParameter());

  // Pass the second through forgetGate.
  forgetGateModule->Forward(output.submat(1 * outSize, 0, 2 * outSize - 1,
      batchSize - 1), forgetGateModule->OutputParameter());

  OutputType modInput = forgetGateModule->OutputParameter() % *prevOutput;

  // Pass that through the outputHidden2GateModule.
  outputHidden2GateModule->Forward(modInput,
      outputHidden2GateModule->OutputParameter());

  // Merge for ot.
  OutputType outputH = input2GateModule->OutputParameter().submat(2 * outSize,
      0, 3 * outSize - 1, batchSize - 1) +
      outputHidden2GateModule->OutputParameter();

  // Pass it through hiddenGate.
  hiddenStateModule->ForwardVisitor(outputH,
      hiddenStateModule->OutputParameter());

  // Update the output (nextOutput): cmul1 + cmul2
  // Where cmul1 is input gate * prevOutput and
  // cmul2 is (1 - input gate) * hidden gate.
  output = (inputGateModule->OutputParameter()
      % (*prevOutput - hiddenStateModule->OutputParameter())) +
      hiddenStateModule->OutputParameter();

  forwardStep++;
  if (forwardStep == rho)
  {
    forwardStep = 0;
    if (this->training)
    {
      outParameter.emplace_back(allZeros.memptr(),
          allZeros.n_rows, allZeros.n_cols, false, true);
      prevOutput = --outParameter.end();
    }
    else
    {
      *prevOutput = arma::mat(allZeros.memptr(),
          allZeros.n_rows, allZeros.n_cols, false, true);
    }
  }
  else if (this->training)
  {
    outParameter.push_back(output);
    prevOutput = --outParameter.end();
  }
  else
  {
    if (forwardStep == 1)
    {
      outParameter.clear();
      outParameter.push_back(output);

      prevOutput = outParameter.begin();
    }
    else
    {
      *prevOutput = output;
    }
  }
}

template<typename InputType, typename OutputType>
void GRU<InputType, OutputType>::Backward(
    const InputType& input, const OutputType& gy, OutputType& g)
{
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    prevError.resize(3 * outSize, batchSize);
    allZeros.zeros(outSize, batchSize);
    // Batch size better not change during an iteration...
    if (outParameter.size() > 1)
    {
      Log::Fatal << "GRU<>::Forward(): batch size cannot change during a "
          << "forward pass!" << std::endl;
    }

    outParameter.clear();
    outParameter.emplace_back(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true);

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  OutputType gyLocal;
  if ((outParameter.size() - backwardStep  - 1) % rho != 0 && backwardStep != 0)
  {
    gyLocal = gy + output2GateModule->Delta();
  }
  else
  {
    gyLocal = OutputType(((OutputType&) gy).memptr(), gy.n_rows, gy.n_cols,
        false, false);
  }

  if (backIterator == outParameter.end())
  {
    backIterator = --(--outParameter.end());
  }

  // Delta zt.
  OutputType dZt = gyLocal % (*backIterator -
      hiddenStateModule->OutputParameter());

  // Delta ot.
  OutputType dOt = gyLocal % (ones<OutputType>(outSize, batchSize) -
      inputGateModule->OutputParameter());

  // Delta of input gate.
  inputGateModule->Backward(inputGateModule->OutputParameter(), dZt,
      inputGateModule->Delta());

  // Delta of hidden gate.
  hiddenStateModule->Backward(hiddenStateModule->OutputParameter(), dOt,
      hiddenStateModule->Delta());

  // Delta of outputHidden2GateModule.
  outputHidden2GateModule->Backward(outputHidden2GateModule->OutputParameter(),
      hiddenStateModule->Delta(), outputHidden2GateModule->Delta());

  // Delta rt.
  OutputType dRt = outputHidden2GateModule->Delta() % *backIterator;

  // Delta of forget gate.
  forgetGateModule->Backward(forgetGateModule->OutputParameter(), dRt,
      forgetGateModule->Delta());

  // Put delta zt.
  prevError.submat(0, 0, 1 * outSize - 1, batchSize - 1) =
      inputGateModule->Delta();

  // Put delta rt.
  prevError.submat(1 * outSize, 0, 2 * outSize - 1, batchSize - 1) =
      forgetGateModule->Delta();

  // Put delta ot.
  prevError.submat(2 * outSize, 0, 3 * outSize - 1, batchSize - 1) =
      hiddenStateModule->Delta();

  // Get delta ht - 1 for input gate and forget gate.
  OutputType prevErrorSubview = prevError.submat(0, 0, 2 * outSize - 1,
      batchSize - 1);
  output2GateModule->Backward(input2GateModule->OutputParameter(),
      prevErrorSubview, output2GateModule->Delta());

  // Add delta ht - 1 from hidden state.
  output2GateModule->Delta() += outputHidden2GateModule->Delta() %
      forgetGateModule->OutputParameter();

  // Add delta ht - 1 from ht.
  output2GateModule->Delta() += gyLocal % inputGateModule->OutputParameter();

  // Get delta input.
  input2GateModule->Backward(input2GateModule->OutputParameter(), prevError,
      input2GateModule->Delta());

  backwardStep++;
  backIterator--;

  g = input2GateModule->Delta();
}

template<typename InputType, typename OutputType>
void GRU<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& /* error */,
    OutputType& /* gradient */)
{
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    prevError.resize(3 * outSize, batchSize);
    allZeros.zeros(outSize, batchSize);
    // Batch size better not change during an iteration...
    if (outParameter.size() > 1)
    {
      Log::Fatal << "GRU<>::Forward(): batch size cannot change during a "
          << "forward pass!" << std::endl;
    }

    outParameter.clear();
    outParameter.emplace_back(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true);

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  if (gradIterator == outParameter.end())
  {
    gradIterator = --(--outParameter.end());
  }

  input2GateModule->Gradient(input, prevError, input2GateModule->Gradient());

  output2GateModule->Gradient(*gradIterator,
      prevError.submat(0, 0, 2 * outSize - 1, batchSize - 1),
      output2GateModule->Gradient());

  outputHidden2GateModule->Gradient(
      *gradIterator % forgetGateModule->OutputParameter(),
      prevError.submat(2 * outSize, 0, 3 * outSize - 1, batchSize - 1),
      outputHidden2GateModule->Gradient());

  gradIterator--;
}

template<typename InputType, typename OutputType>
void GRU<InputType, OutputType>::ResetCell(const size_t /* size */)
{
  outParameter.clear();
  outParameter.emplace_back(allZeros.memptr(),
    allZeros.n_rows, allZeros.n_cols, false, true);

  prevOutput = outParameter.begin();
  backIterator = outParameter.end();
  gradIterator = outParameter.end();

  forwardStep = 0;
  backwardStep = 0;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void GRU<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  // If necessary, clean memory from the old model.
  // TODO: CEREAL_POINTER() should clean memory automatically...

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(rho));

  ar(CEREAL_NVP(weights));

  ar(CEREAL_POINTER(input2GateModule));
  ar(CEREAL_POINTER(output2GateModule));
  ar(CEREAL_POINTER(outputHidden2GateModule));
  ar(CEREAL_POINTER(inputGateModule));
  ar(CEREAL_POINTER(forgetGateModule));
  ar(CEREAL_POINTER(hiddenStateModule));
}

} // namespace mlpack

#endif
