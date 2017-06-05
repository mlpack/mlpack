/**
 * @file lstm_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LSTM class, which implements a lstm network
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
#include "linear.hpp"

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
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false)
{
  // input specific linear layers(for zt, rt, ot)
  input2GateModule = new Linear<>(inSize, 3 * outSize);
  
  // zt and rt
  output2GateModule = new LinearNoBias<>(outSize, 2 * outSize);
  
  // output for ot
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

  prevOutput = arma::zeros<arma::mat>(outSize, 1);
  prevError = arma::zeros<arma::mat>(3 * outSize, 1);

  outParameter.reserve(rho);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GRU<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  if (!deterministic)
  {
    outParameter.push_back(prevOutput);
  }
  
  // process the input linearly(zt, rt, ot)
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, input2GateModule))),
      input2GateModule);

  // process the output(zt, rt) linearly
  boost::apply_visitor(ForwardVisitor(std::move(prevOutput), std::move(
      boost::apply_visitor(outputParameterVisitor, output2GateModule))),
      output2GateModule);
      
  // merge the outputs(zt and rt)
  output = (boost::apply_visitor(outputParameterVisitor, 
      input2GateModule).submat(0, 0, 2 * outSize - 1, 0) +
      boost::apply_visitor(outputParameterVisitor, output2GateModule));

  // pass the first outSize through inputGate
  boost::apply_visitor(ForwardVisitor(std::move(output.submat(
      0, 0, 1 * outSize - 1, 0)), std::move(boost::apply_visitor(
      outputParameterVisitor, inputGateModule))), inputGateModule);
      
  // pass the second through forgetGate
  boost::apply_visitor(ForwardVisitor(std::move(output.submat(
      1 * outSize, 0, 2 * outSize - 1, 0)), std::move(boost::apply_visitor(
      outputParameterVisitor, forgetGateModule))), forgetGateModule);
      
  //temp = output of forgetGate(rt) % prevOutput
  arma::mat modInput = (boost::apply_visitor(outputParameterVisitor, 
      forgetGateModule) % prevOutput);
      
  // pass that through the outputHidden2GateModule
  boost::apply_visitor(ForwardVisitor(std::move(modInput), std::move(
      boost::apply_visitor(outputParameterVisitor, outputHidden2GateModule))),
      outputHidden2GateModule);
  
  // merge them
  arma::mat outputH = boost::apply_visitor(outputParameterVisitor, 
      input2GateModule).submat(2 * outSize, 0, 3 * outSize - 1, 0) + 
      boost::apply_visitor(outputParameterVisitor, outputHidden2GateModule);
  
  // pass it through hiddenGate
  boost::apply_visitor(ForwardVisitor(std::move(outputH), std::move(
      boost::apply_visitor(outputParameterVisitor, hiddenStateModule))), 
      hiddenStateModule);

  // Update the output (nextOutput): cmul1 + cmul2
  // where cmul1 is input gate * prevOutput and
  // cmul2 is (1 - input gate) * hidden gate.
  output = (boost::apply_visitor(outputParameterVisitor, inputGateModule) 
      % prevOutput) + 
      ((arma::ones<arma::vec>(outSize) - 
      boost::apply_visitor(outputParameterVisitor, inputGateModule)) % 
      boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule));
      
  prevOutput = output;

  forwardStep++;
  if (forwardStep == rho)
  {
    forwardStep = 0;
    prevOutput.zeros();
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GRU<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  if (backwardStep > 0)
  {
    gy += boost::apply_visitor(deltaVisitor, output2GateModule);
  }
  
  // delta zt
  arma::mat d_zt = gy % (outParameter[outParameter.size() -
      backwardStep - 1] - boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule));
      
  // delta ot
  arma::mat d_ot = gy % (arma::ones<arma::vec>(outSize) - 
      boost::apply_visitor(outputParameterVisitor, inputGateModule));
      
  // delta of input gate
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, inputGateModule)), std::move(d_zt),
      std::move(boost::apply_visitor(deltaVisitor, inputGateModule))),
      inputGateModule);
      
  // delta of hidden gate
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, hiddenStateModule)), std::move(d_ot),
      std::move(boost::apply_visitor(deltaVisitor, hiddenStateModule))),
      hiddenStateModule);
  
  // delta of outputHidden2GateModule
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, outputHidden2GateModule)), 
      std::move(boost::apply_visitor(deltaVisitor, hiddenStateModule)),
      std::move(boost::apply_visitor(deltaVisitor, outputHidden2GateModule))),
      outputHidden2GateModule);
      
  // delta rt
  arma::mat d_rt = boost::apply_visitor(deltaVisitor, outputHidden2GateModule) %
      outParameter[outParameter.size() - backwardStep - 1];
      
  // delta of forget gate
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, forgetGateModule)), std::move(d_rt),
      std::move(boost::apply_visitor(deltaVisitor, forgetGateModule))),
      forgetGateModule);
  
  // put delta zt
  prevError.submat(0, 0, 1 * outSize - 1, 0) = boost::apply_visitor(
      deltaVisitor, inputGateModule);
  
  // put delta rt
  prevError.submat(1 * outSize, 0, 2 * outSize - 1, 0) = boost::apply_visitor(
      deltaVisitor, forgetGateModule);
  
  // put delta ot
  prevError.submat(2 * outSize, 0, 3 * outSize - 1, 0) = boost::apply_visitor(
      deltaVisitor, hiddenStateModule);
      
  // get delta ht - 1 for input gate and forget gate
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, input2GateModule)), 
      std::move(prevError.submat(0, 0, 2 * outSize - 1, 0)),
      std::move(boost::apply_visitor(deltaVisitor, output2GateModule))),
      output2GateModule);   
  
  // add delta ht - 1 from hidden state
  boost::apply_visitor(deltaVisitor, output2GateModule) += 
      boost::apply_visitor(deltaVisitor, outputHidden2GateModule) %
      boost::apply_visitor(outputParameterVisitor, forgetGateModule);
      
  // add delta ht - 1 from ht
  boost::apply_visitor(deltaVisitor, output2GateModule) += gy %
      boost::apply_visitor(outputParameterVisitor, inputGateModule);
      
  // get delta input
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, input2GateModule)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, input2GateModule))),
      input2GateModule);
      
  backwardStep++;
  if (backwardStep == rho)
  {
    backwardStep = 0;
  }

  g = boost::apply_visitor(deltaVisitor, input2GateModule);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GRU<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(std::move(input), std::move(prevError)),
      input2GateModule);

  boost::apply_visitor(GradientVisitor(
      std::move(outParameter[outParameter.size() - gradientStep - 1]),
      std::move(prevError.submat(0, 0, 2 * outSize - 1, 0))), 
      output2GateModule);
      
  boost::apply_visitor(GradientVisitor(
      std::move(outParameter[outParameter.size() - gradientStep - 1]),
      std::move(prevError.submat(2 * outSize, 0, 3 * outSize - 1, 0))), 
      outputHidden2GateModule);

  gradientStep++;
  if (gradientStep == rho)
  {
    gradientStep = 0;
    outParameter.clear();
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void GRU<InputDataType, OutputDataType>::Serialize(
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
