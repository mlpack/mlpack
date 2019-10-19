/**
 * @file gru_impl.hpp
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

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
GRU<InputDataType, OutputDataType>::GRU()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
GRU<InputDataType, OutputDataType>::GRU(
    const size_t inSize,
    const size_t outSize,
    const size_t rho) :
    inSize(inSize),
    outSize(outSize),
    rho(rho),
    batchSize(1),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false)
{
  // Input specific linear layers(for zt, rt, ot).
  input2GateModule = new Linear<>(inSize, 3 * outSize);

  // Previous output gates (for zt and rt).
  output2GateModule = new LinearNoBias<>(outSize, 2 * outSize);

  // Previous output gate for ot.
  outputHidden2GateModule = new LinearNoBias<>(outSize, outSize);

  network.push_back(input2GateModule);
  network.push_back(output2GateModule);
  network.push_back(outputHidden2GateModule);

  inputGateModule = new SigmoidLayer<>();
  forgetGateModule = new SigmoidLayer<>();
  hiddenStateModule = new TanHLayer<>();

  network.push_back(inputGateModule);
  network.push_back(hiddenStateModule);
  network.push_back(forgetGateModule);

  prevError = arma::zeros<arma::mat>(3 * outSize, batchSize);

  allZeros = arma::zeros<arma::mat>(outSize, batchSize);

  outParameter.push_back(std::move(arma::mat(allZeros.memptr(),
      allZeros.n_rows, allZeros.n_cols, false, true)));

  prevOutput = outParameter.begin();
  backIterator = outParameter.end();
  gradIterator = outParameter.end();
}

template<typename InputDataType, typename OutputDataType>
GRU<InputDataType, OutputDataType>::~GRU()
{
  boost::apply_visitor(deleteVisitor, input2GateModule);
  boost::apply_visitor(deleteVisitor, output2GateModule);
  boost::apply_visitor(deleteVisitor, outputHidden2GateModule);
  boost::apply_visitor(deleteVisitor, inputGateModule);
  boost::apply_visitor(deleteVisitor, forgetGateModule);
  boost::apply_visitor(deleteVisitor, hiddenStateModule);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GRU<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
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
    outParameter.push_back(std::move(arma::mat(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true)));

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  // Process the input linearly(zt, rt, ot).
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, input2GateModule))),
      input2GateModule);

  // Process the output(zt, rt) linearly.
  boost::apply_visitor(ForwardVisitor(std::move(*prevOutput), std::move(
      boost::apply_visitor(outputParameterVisitor, output2GateModule))),
      output2GateModule);

  // Merge the outputs(zt and rt).
  output = (boost::apply_visitor(outputParameterVisitor,
      input2GateModule).submat(0, 0, 2 * outSize - 1, batchSize - 1) +
      boost::apply_visitor(outputParameterVisitor, output2GateModule));

  // Pass the first outSize through inputGate(it).
  boost::apply_visitor(ForwardVisitor(std::move(output.submat(
      0, 0, 1 * outSize - 1, batchSize - 1)), std::move(boost::apply_visitor(
      outputParameterVisitor, inputGateModule))), inputGateModule);

  // Pass the second through forgetGate.
  boost::apply_visitor(ForwardVisitor(std::move(output.submat(
      1 * outSize, 0, 2 * outSize - 1, batchSize - 1)), std::move(
      boost::apply_visitor(outputParameterVisitor, forgetGateModule))),
      forgetGateModule);

  arma::mat modInput = (boost::apply_visitor(outputParameterVisitor,
      forgetGateModule) % *prevOutput);

  // Pass that through the outputHidden2GateModule.
  boost::apply_visitor(ForwardVisitor(std::move(modInput), std::move(
      boost::apply_visitor(outputParameterVisitor, outputHidden2GateModule))),
      outputHidden2GateModule);

  // Merge for ot.
  arma::mat outputH = boost::apply_visitor(outputParameterVisitor,
      input2GateModule).submat(2 * outSize, 0, 3 * outSize - 1, batchSize - 1) +
      boost::apply_visitor(outputParameterVisitor, outputHidden2GateModule);

  // Pass it through hiddenGate.
  boost::apply_visitor(ForwardVisitor(std::move(outputH), std::move(
      boost::apply_visitor(outputParameterVisitor, hiddenStateModule))),
      hiddenStateModule);

  // Update the output (nextOutput): cmul1 + cmul2
  // Where cmul1 is input gate * prevOutput and
  // cmul2 is (1 - input gate) * hidden gate.
  output = (boost::apply_visitor(outputParameterVisitor, inputGateModule)
      % (*prevOutput - boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule))) + boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule);

  forwardStep++;
  if (forwardStep == rho)
  {
    forwardStep = 0;
    if (!deterministic)
    {
      outParameter.push_back(std::move(arma::mat(allZeros.memptr(),
          allZeros.n_rows, allZeros.n_cols, false, true)));
      prevOutput = --outParameter.end();
    }
    else
    {
      *prevOutput = std::move(arma::mat(allZeros.memptr(),
          allZeros.n_rows, allZeros.n_cols, false, true));
    }
  }
  else if (!deterministic)
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

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GRU<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& input, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
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
    outParameter.push_back(std::move(arma::mat(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true)));

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  if ((outParameter.size() - backwardStep  - 1) % rho != 0 && backwardStep != 0)
  {
    gy += boost::apply_visitor(deltaVisitor, output2GateModule);
  }

  if (backIterator == outParameter.end())
  {
    backIterator = --(--outParameter.end());
  }

  // Delta zt.
  arma::mat dZt = gy % (*backIterator -
      boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule));

  // Delta ot.
  arma::mat dOt = gy % (arma::ones<arma::mat>(outSize, batchSize) -
      boost::apply_visitor(outputParameterVisitor, inputGateModule));

  // Delta of input gate.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, inputGateModule)), std::move(dZt),
      std::move(boost::apply_visitor(deltaVisitor, inputGateModule))),
      inputGateModule);

  // Delta of hidden gate.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, hiddenStateModule)), std::move(dOt),
      std::move(boost::apply_visitor(deltaVisitor, hiddenStateModule))),
      hiddenStateModule);

  // Delta of outputHidden2GateModule.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, outputHidden2GateModule)),
      std::move(boost::apply_visitor(deltaVisitor, hiddenStateModule)),
      std::move(boost::apply_visitor(deltaVisitor, outputHidden2GateModule))),
      outputHidden2GateModule);

  // Delta rt.
  arma::mat dRt = boost::apply_visitor(deltaVisitor, outputHidden2GateModule) %
      *backIterator;

  // Delta of forget gate.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, forgetGateModule)), std::move(dRt),
      std::move(boost::apply_visitor(deltaVisitor, forgetGateModule))),
      forgetGateModule);

  // Put delta zt.
  prevError.submat(0, 0, 1 * outSize - 1, batchSize - 1) = boost::apply_visitor(
      deltaVisitor, inputGateModule);

  // Put delta rt.
  prevError.submat(1 * outSize, 0, 2 * outSize - 1, batchSize - 1) =
      boost::apply_visitor(deltaVisitor, forgetGateModule);

  // Put delta ot.
  prevError.submat(2 * outSize, 0, 3 * outSize - 1, batchSize - 1) =
      boost::apply_visitor(deltaVisitor, hiddenStateModule);

  // Get delta ht - 1 for input gate and forget gate.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, input2GateModule)),
      std::move(prevError.submat(0, 0, 2 * outSize - 1, batchSize - 1)),
      std::move(boost::apply_visitor(deltaVisitor, output2GateModule))),
      output2GateModule);

  // Add delta ht - 1 from hidden state.
  boost::apply_visitor(deltaVisitor, output2GateModule) +=
      boost::apply_visitor(deltaVisitor, outputHidden2GateModule) %
      boost::apply_visitor(outputParameterVisitor, forgetGateModule);

  // Add delta ht - 1 from ht.
  boost::apply_visitor(deltaVisitor, output2GateModule) += gy %
      boost::apply_visitor(outputParameterVisitor, inputGateModule);

  // Get delta input.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, input2GateModule)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, input2GateModule))),
      input2GateModule);

  backwardStep++;
  backIterator--;

  g = boost::apply_visitor(deltaVisitor, input2GateModule);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GRU<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
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
    outParameter.push_back(std::move(arma::mat(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true)));

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  if (gradIterator == outParameter.end())
  {
    gradIterator = --(--outParameter.end());
  }

  boost::apply_visitor(GradientVisitor(std::move(input), std::move(prevError)),
      input2GateModule);

  boost::apply_visitor(GradientVisitor(
      std::move(*gradIterator),
      std::move(prevError.submat(0, 0, 2 * outSize - 1, batchSize - 1))),
      output2GateModule);

  boost::apply_visitor(GradientVisitor(
      *gradIterator % boost::apply_visitor(outputParameterVisitor,
      forgetGateModule),
      std::move(prevError.submat(2 * outSize, 0, 3 * outSize - 1,
      batchSize - 1))), outputHidden2GateModule);

  gradIterator--;
}

template<typename InputDataType, typename OutputDataType>
void GRU<InputDataType, OutputDataType>::ResetCell(const size_t /* size */)
{
  outParameter.clear();
  outParameter.push_back(std::move(arma::mat(allZeros.memptr(),
    allZeros.n_rows, allZeros.n_cols, false, true)));

  prevOutput = outParameter.begin();
  backIterator = outParameter.end();
  gradIterator = outParameter.end();

  forwardStep = 0;
  backwardStep = 0;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void GRU<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  // If necessary, clean memory from the old model.
  if (Archive::is_loading::value)
  {
    boost::apply_visitor(deleteVisitor, input2GateModule);
    boost::apply_visitor(deleteVisitor, output2GateModule);
    boost::apply_visitor(deleteVisitor, outputHidden2GateModule);
    boost::apply_visitor(deleteVisitor, inputGateModule);
    boost::apply_visitor(deleteVisitor, forgetGateModule);
    boost::apply_visitor(deleteVisitor, hiddenStateModule);
  }

  ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);
  ar & BOOST_SERIALIZATION_NVP(rho);

  ar & BOOST_SERIALIZATION_NVP(input2GateModule);
  ar & BOOST_SERIALIZATION_NVP(output2GateModule);
  ar & BOOST_SERIALIZATION_NVP(outputHidden2GateModule);
  ar & BOOST_SERIALIZATION_NVP(inputGateModule);
  ar & BOOST_SERIALIZATION_NVP(forgetGateModule);
  ar & BOOST_SERIALIZATION_NVP(hiddenStateModule);
}

} // namespace ann
} // namespace mlpack

#endif
