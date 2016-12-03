/**
 * @file lstm.hpp
 * @author Marcus Edel
 *
 * Definition of the LSTM class, which implements a lstm network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LSTM_HPP
#define MLPACK_METHODS_ANN_LAYER_LSTM_HPP

#include <mlpack/core.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include "layer_types.hpp"
#include "add_merge.hpp"
#include "sequential.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a lstm network layer.
 *
 * This class allows specification of the type of the activation functions used
 * for the gates and cells and also of the type of the function used to
 * initialize and update the peephole weights.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class LSTM
{
 public:
  //! Create the LSTM object.
<<<<<<< HEAD
  LSTM();
=======
  LSTM() { /* Nothing to do here */ }
>>>>>>> Refactor ann layer.

  /**
   * Create the LSTM layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
<<<<<<< HEAD
  LSTM(const size_t inSize, const size_t outSize, const size_t rho);
=======
  LSTM(const size_t inSize, const size_t outSize, const size_t rho) :
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
>>>>>>> Refactor ann layer.

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
<<<<<<< HEAD
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output);
=======
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    if (!deterministic)
    {
      cellParameter.push_back(prevCell);
      outParameter.push_back(prevOutput);
    }

    arma::mat output1;
    arma::mat output2;
    arma::mat output3;

    boost::apply_visitor(
      ForwardVisitor(
        std::move(input),
        std::move(boost::apply_visitor(outputParameterVisitor,
            input2GateModule))
      ),
      input2GateModule);

    boost::apply_visitor(
      ForwardVisitor(
        std::move(prevOutput),
        std::move(boost::apply_visitor(outputParameterVisitor,
            output2GateModule))
      ),
      output2GateModule);

    output = boost::apply_visitor(outputParameterVisitor, input2GateModule) +
        boost::apply_visitor(outputParameterVisitor, output2GateModule);

    boost::apply_visitor(
      ForwardVisitor(
        std::move(output.submat(0, 0, 1 * outSize - 1, 0)),
        std::move(boost::apply_visitor(outputParameterVisitor,
            inputGateModule))
      ),
      inputGateModule);

    boost::apply_visitor(
      ForwardVisitor(
        std::move(output.submat(1 * outSize, 0, 2 * outSize - 1, 0)),
        std::move(boost::apply_visitor(outputParameterVisitor,
            hiddenStateModule))
      ),
      hiddenStateModule);

    boost::apply_visitor(
      ForwardVisitor(
        std::move(output.submat(2 * outSize, 0, 3 * outSize - 1, 0)),
        std::move(boost::apply_visitor(outputParameterVisitor,
            forgetGateModule))
      ),
      forgetGateModule);

    boost::apply_visitor(
      ForwardVisitor(
        std::move(output.submat(3 * outSize, 0, 4 * outSize - 1, 0)),
        std::move(boost::apply_visitor(outputParameterVisitor,
            outputGateModule))
      ),
      outputGateModule);

    arma::mat cell = prevCell;

    // Input gate * hidden state.
    arma::mat cmul1 = boost::apply_visitor(outputParameterVisitor,
        inputGateModule) % boost::apply_visitor(outputParameterVisitor,
        hiddenStateModule);

    // Forget gate * cell.
    arma::mat cmul2 = boost::apply_visitor(outputParameterVisitor,
        forgetGateModule) % cell;

    arma::mat nextCell = cmul1 + cmul2;

    boost::apply_visitor(
      ForwardVisitor(
        std::move(nextCell),
        std::move(boost::apply_visitor(outputParameterVisitor, cellModule))
      ),
      cellModule);

    boost::apply_visitor(
      ForwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, cellModule)),
        std::move(boost::apply_visitor(outputParameterVisitor,
            cellActivationModule))
      ),
      cellActivationModule);

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
>>>>>>> Refactor ann layer.

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& gy,
<<<<<<< HEAD
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    if (backwardStep > 0)
    {
      gy += boost::apply_visitor(deltaVisitor, output2GateModule);
    }

    arma::mat g1 = boost::apply_visitor(outputParameterVisitor,
        cellActivationModule) % gy;

    arma::mat g2 = boost::apply_visitor(outputParameterVisitor,
        outputGateModule) % gy;

    boost::apply_visitor(
      BackwardVisitor(
            std::move(boost::apply_visitor(outputParameterVisitor,
                cellActivationModule)),
            std::move(g2),
            std::move(boost::apply_visitor(deltaVisitor,
                cellActivationModule))
      ),
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

    boost::apply_visitor(
      BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
            inputGateModule)),
        std::move(g5),
        std::move(boost::apply_visitor(deltaVisitor, inputGateModule))
    ),
    inputGateModule);

    boost::apply_visitor(
      BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
            hiddenStateModule)),
        std::move(g4),
        std::move(boost::apply_visitor(deltaVisitor, hiddenStateModule))
    ),
    hiddenStateModule);

    boost::apply_visitor(
      BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
            forgetGateModule)),
        std::move(g7),
        std::move(boost::apply_visitor(deltaVisitor, forgetGateModule))
    ),
    forgetGateModule);

    boost::apply_visitor(
      BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
            outputGateModule)),
        std::move(g1),
        std::move(boost::apply_visitor(deltaVisitor, outputGateModule))
    ),
    outputGateModule);

    prevError.submat(0, 0, 1 * outSize - 1, 0) = boost::apply_visitor(
        deltaVisitor, inputGateModule);
    prevError.submat(1 * outSize, 0, 2 * outSize - 1, 0) = boost::apply_visitor(
        deltaVisitor, hiddenStateModule);
    prevError.submat(2 * outSize, 0, 3 * outSize - 1, 0) = boost::apply_visitor(
        deltaVisitor, forgetGateModule);
    prevError.submat(3 * outSize, 0, 4 * outSize - 1, 0) = boost::apply_visitor(
        deltaVisitor, outputGateModule);

    boost::apply_visitor(
      BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
            input2GateModule)),
        std::move(prevError),
        std::move(boost::apply_visitor(deltaVisitor, input2GateModule))
    ),
    input2GateModule);

    boost::apply_visitor(
      BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
            output2GateModule)),
        std::move(prevError),
        std::move(boost::apply_visitor(deltaVisitor, output2GateModule))
    ),
    output2GateModule);

    backwardStep++;
    if (backwardStep == rho)
    {
      backwardStep = 0;
      cellParameter.clear();
    }

    g = boost::apply_visitor(deltaVisitor, input2GateModule);
  }
>>>>>>> Refactor ann layer.

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Mat<eT>&& input,
                arma::Mat<eT>&& /* error */,
<<<<<<< HEAD
                arma::Mat<eT>&& /* gradient */);
=======
                arma::Mat<eT>&& /* gradient */)
  {
    boost::apply_visitor(
        GradientVisitor(
          std::move(input),
          std::move(prevError)
      ),
      input2GateModule);

    boost::apply_visitor(
        GradientVisitor(
          std::move(outParameter[outParameter.size() - gradientStep - 1]),
          std::move(prevError)
      ),
      output2GateModule);

    gradientStep++;
    if (gradientStep == rho)
    {
      gradientStep = 0;
      outParameter.clear();
    }
  }
>>>>>>> Refactor ann layer.

  //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the maximum number of steps to backpropagate through time (BPTT).
  size_t Rho() const { return rho; }
  //! Modify the maximum number of steps to backpropagate through time (BPTT).
  size_t& Rho() { return rho; }

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! Get the model modules.
  std::vector<LayerTypes>& Model() { return network; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
<<<<<<< HEAD
  void Serialize(Archive& ar, const unsigned int /* version */);
=======
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(weights, "weights");
    ar & data::CreateNVP(inSize, "inSize");
    ar & data::CreateNVP(outSize, "outSize");
    ar & data::CreateNVP(rho, "rho");
  }
>>>>>>> Refactor ann layer.

 private:

  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Number of steps to backpropagate through time (BPTT).
  size_t rho;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored previous output.
  arma::mat prevOutput;

  //! Locally-stored previous cell state.
  arma::mat prevCell;

  //! Locally-stored input 2 gate module.
  LayerTypes input2GateModule;

  //! Locally-stored output 2 gate module.
  LayerTypes output2GateModule;

  //! Locally-stored input gate module.
  LayerTypes inputGateModule;

  //! Locally-stored hidden state module.
  LayerTypes hiddenStateModule;

  //! Locally-stored forget gate module.
  LayerTypes forgetGateModule;

  //! Locally-stored output gate module.
  LayerTypes outputGateModule;

  //! Locally-stored cell module.
  LayerTypes cellModule;

  //! Locally-stored cell activation module.
  LayerTypes cellActivationModule;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored list of network modules.
  std::vector<LayerTypes> network;

  //! Locally-stored number of forward steps.
  size_t forwardStep;

  //! Locally-stored number of backward steps.
  size_t backwardStep;

  //! Locally-stored number of gradient steps.
  size_t gradientStep;

  //! Locally-stored cell parameters.
  std::vector<arma::mat> cellParameter;

  //! Locally-stored output parameters.
  std::vector<arma::mat> outParameter;

  //! Locally-stored previous error.
  arma::mat prevError;

  //! Locally-stored cell activation error.
  arma::mat cellActivationError;

  //! Locally-stored foget gate error.
  arma::mat forgetGateError;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class LSTM

} // namespace ann
} // namespace mlpack

<<<<<<< HEAD
// Include implementation.
#include "lstm_impl.hpp"

=======
>>>>>>> Refactor ann layer.
#endif
