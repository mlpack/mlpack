/**
 * @file lstm_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the LSTM class, which implements a lstm network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LSTM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LSTM_IMPL_HPP

// In case it hasn't yet been included.
#include "linear.hpp"

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
LSTM<InputDataType, OutputDataType>::LSTM()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
LSTM<InputDataType, OutputDataType>::LSTM(
    const size_t inSize,
    const size_t outSize,
    const size_t rho) :
    inSize(inSize),
    outSize(outSize),
    rho(rho),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false)
{
  input2GateModule = new Linear<>(inSize, 4 * outSize);
  output2GateModule = new LinearNoBias<>(outSize, 4 * outSize);

  network.push_back(input2GateModule);
  network.push_back(output2GateModule);

  inputGateModule = new SigmoidLayer<>();
  hiddenStateModule = new TanHLayer<>();
  forgetGateModule = new SigmoidLayer<>();
  outputGateModule = new SigmoidLayer<>();

  network.push_back(inputGateModule);
  network.push_back(hiddenStateModule);
  network.push_back(forgetGateModule);
  network.push_back(outputGateModule);

  cellModule = new IdentityLayer<>();
  cellActivationModule = new TanHLayer<>();

  network.push_back(cellModule);
  network.push_back(cellActivationModule);

  prevOutput = arma::zeros<arma::mat>(outSize, 1);
  prevCell = arma::zeros<arma::mat>(outSize, 1);
  prevError = arma::zeros<arma::mat>(4 * outSize, 1);
  cellActivationError = arma::zeros<arma::mat>(outSize, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void LSTM<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  if (!deterministic)
  {
    cellParameter.push_back(prevCell);
    outParameter.push_back(prevOutput);
  }

  arma::mat output1;
  arma::mat output2;
  arma::mat output3;

  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, input2GateModule))),
      input2GateModule);

  boost::apply_visitor(ForwardVisitor(std::move(prevOutput), std::move(
      boost::apply_visitor(outputParameterVisitor, output2GateModule))),
      output2GateModule);

  output = boost::apply_visitor(outputParameterVisitor, input2GateModule) +
      boost::apply_visitor(outputParameterVisitor, output2GateModule);

  boost::apply_visitor(ForwardVisitor(std::move(output.submat(
      0, 0, 1 * outSize - 1, 0)), std::move(boost::apply_visitor(
      outputParameterVisitor, inputGateModule))), inputGateModule);

  boost::apply_visitor(ForwardVisitor(std::move(output.submat(
      1 * outSize, 0, 2 * outSize - 1, 0)), std::move(boost::apply_visitor(
      outputParameterVisitor, hiddenStateModule))), hiddenStateModule);

  boost::apply_visitor(ForwardVisitor(std::move(output.submat(
      2 * outSize, 0, 3 * outSize - 1, 0)), std::move(boost::apply_visitor(
      outputParameterVisitor, forgetGateModule))), forgetGateModule);

  boost::apply_visitor(ForwardVisitor(std::move(output.submat(
      3 * outSize, 0, 4 * outSize - 1, 0)), std::move(boost::apply_visitor(
      outputParameterVisitor, outputGateModule))), outputGateModule);

  arma::mat cell = prevCell;

  // Input gate * hidden state.
  arma::mat cmul1 = boost::apply_visitor(outputParameterVisitor,
      inputGateModule) % boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule);

  // Forget gate * cell.
  arma::mat cmul2 = boost::apply_visitor(outputParameterVisitor,
      forgetGateModule) % cell;

  arma::mat nextCell = cmul1 + cmul2;

  boost::apply_visitor(ForwardVisitor(std::move(nextCell), std::move(
    boost::apply_visitor(outputParameterVisitor, cellModule))), cellModule);

  boost::apply_visitor(ForwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, cellModule)), std::move(boost::apply_visitor(
      outputParameterVisitor, cellActivationModule))), cellActivationModule);

  output = boost::apply_visitor(outputParameterVisitor,
      cellActivationModule) % boost::apply_visitor(outputParameterVisitor,
      outputGateModule);

  prevCell = nextCell;
  prevOutput = output;

  forwardStep++;
  if (forwardStep == rho)
  {
    forwardStep = 0;
    prevOutput.zeros();
    prevCell.zeros();
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void LSTM<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  if (backwardStep > 0)
  {
    gy += boost::apply_visitor(deltaVisitor, output2GateModule);
  }

  arma::mat g1 = boost::apply_visitor(outputParameterVisitor,
      cellActivationModule) % gy;

  arma::mat g2 = boost::apply_visitor(outputParameterVisitor,
      outputGateModule) % gy;

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, cellActivationModule)), std::move(g2),
      std::move(boost::apply_visitor(deltaVisitor, cellActivationModule))),
      cellActivationModule);

  cellActivationError = boost::apply_visitor(deltaVisitor,
      cellActivationModule);

  if (backwardStep > 0)
  {
    cellActivationError += forgetGateError;
  }

  arma::mat g4 = boost::apply_visitor(outputParameterVisitor,
      inputGateModule) % cellActivationError;

  arma::mat g5 = boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule) % cellActivationError;

  forgetGateError = boost::apply_visitor(outputParameterVisitor,
      forgetGateModule) % cellActivationError;

  arma::mat g7 = cellParameter[cellParameter.size() -
      backwardStep - 1] % cellActivationError;

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, inputGateModule)), std::move(g5),
      std::move(boost::apply_visitor(deltaVisitor, inputGateModule))),
      inputGateModule);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, hiddenStateModule)), std::move(g4),
      std::move(boost::apply_visitor(deltaVisitor, hiddenStateModule))),
      hiddenStateModule);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, forgetGateModule)), std::move(g7),
      std::move(boost::apply_visitor(deltaVisitor, forgetGateModule))),
      forgetGateModule);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, outputGateModule)), std::move(g1),
      std::move(boost::apply_visitor(deltaVisitor, outputGateModule))),
      outputGateModule);

  prevError.submat(0, 0, 1 * outSize - 1, 0) = boost::apply_visitor(
      deltaVisitor, inputGateModule);
  prevError.submat(1 * outSize, 0, 2 * outSize - 1, 0) = boost::apply_visitor(
      deltaVisitor, hiddenStateModule);
  prevError.submat(2 * outSize, 0, 3 * outSize - 1, 0) = boost::apply_visitor(
      deltaVisitor, forgetGateModule);
  prevError.submat(3 * outSize, 0, 4 * outSize - 1, 0) = boost::apply_visitor(
      deltaVisitor, outputGateModule);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, input2GateModule)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, input2GateModule))),
      input2GateModule);

  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, output2GateModule)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, output2GateModule))),
      output2GateModule);

  backwardStep++;
  if (backwardStep == rho)
  {
    backwardStep = 0;
    cellParameter.clear();
  }

  g = boost::apply_visitor(deltaVisitor, input2GateModule);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void LSTM<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(std::move(input), std::move(prevError)),
      input2GateModule);

  boost::apply_visitor(GradientVisitor(
      std::move(outParameter[outParameter.size() - gradientStep - 1]),
      std::move(prevError)), output2GateModule);

  gradientStep++;
  if (gradientStep == rho)
  {
    gradientStep = 0;
    outParameter.clear();
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void LSTM<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(weights, "weights");
  ar & data::CreateNVP(inSize, "inSize");
  ar & data::CreateNVP(outSize, "outSize");
  ar & data::CreateNVP(rho, "rho");
}

} // namespace ann
} // namespace mlpack

#endif
