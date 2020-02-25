/**
 * @file lstm.hpp
 * @author Marcus Edel
 *
 * Definition of the LSTM class, which implements a LSTM network layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LSTM_HPP
#define MLPACK_METHODS_ANN_LAYER_LSTM_HPP

#include <mlpack/prereqs.hpp>
#include <limits>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the LSTM module class.
 * The implementation corresponds to the following algorithm:
 *
 * @f{eqnarray}{
 * i &=& sigmoid(W \cdot x + W \cdot h + W \cdot c + b) \\
 * f &=& sigmoid(W  \cdot x + W \cdot h + W \cdot c + b) \\
 * z &=& tanh(W \cdot x + W \cdot h + b) \\
 * c &=& f \cdot c + i \cdot z \\
 * o &=& sigmoid(W \cdot x + W \cdot h + W \cdot c + b) \\
 * h &=& o \cdot tanh(c)
 * @f}
 *
 * Note that if an LSTM layer is desired as the first layer of a neural network,
 * an IdentityLayer should be added to the network as the first layer, and then
 * the LSTM layer should be added.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Graves2013,
 *   author  = {Alex Graves and Abdel{-}rahman Mohamed and Geoffrey E. Hinton},
 *   title   = {Speech Recognition with Deep Recurrent Neural Networks},
 *   journal = CoRR},
 *   year    = {2013},
 *   url     = {http://arxiv.org/abs/1303.5778},
 * }
 * @endcode
 *
 * \see FastLSTM for a faster LSTM version which combines the calculation of the
 * input, forget, output gates and hidden state in a single step.
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
  LSTM();

  /**
   * Create the LSTM layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  LSTM(const size_t inSize,
       const size_t outSize,
       const size_t rho = std::numeric_limits<size_t>::max());

  /**
   * Ordinary feed-forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(InputType&& input, OutputType&& output);

  /**
   * Ordinary feed-forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   * @param cellState Cell state of the LSTM.
   * @param useCellState Use the cellState passed in the LSTM cell.
   */
  template<typename InputType, typename OutputType>
  void Forward(InputType&& input,
               OutputType&& output,
               OutputType&& cellState,
               bool useCellState = false);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename ErrorType, typename GradientType>
  void Backward(const InputType&& input,
                ErrorType&& gy,
                GradientType&& g);

  /*
   * Reset the layer parameter.
   */
  void Reset();

  /*
   * Resets the cell to accept a new input. This breaks the BPTT chain starts a
   * new one.
   *
   * @param size The current maximum number of steps through time.
   */
  void ResetCell(const size_t size);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename InputType, typename ErrorType, typename GradientType>
  void Gradient(InputType&& input,
                ErrorType&& error,
                GradientType&& gradient);

  //! Get the maximum number of steps to backpropagate through time (BPTT).
  size_t Rho() const { return rho; }
  //! Modify the maximum number of steps to backpropagate through time (BPTT).
  size_t& Rho() { return rho; }

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return grad; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return grad; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Number of steps to backpropagate through time (BPTT).
  size_t rho;

  //! Locally-stored number of forward steps.
  size_t forwardStep;

  //! Locally-stored number of backward steps.
  size_t backwardStep;

  //! Locally-stored number of gradient steps.
  size_t gradientStep;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored previous output.
  OutputDataType prevOutput;

  //! Locally-stored batch size.
  size_t batchSize;

  //! Current batch step, alias for batchSize - 1.
  size_t batchStep;

  //! Current gradient step to keep track of the backpropagate through time
  //! step.
  size_t gradientStepIdx;

  //! Locally-stored cell activation error.
  OutputDataType cellActivationError;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType grad;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Weights between the output and input gate.
  OutputDataType output2GateInputWeight;

  //! Weights between the input and gate.
  OutputDataType input2GateInputWeight;

  //! Bias between the input and input gate.
  OutputDataType input2GateInputBias;

  //! Weights between the cell and input gate.
  OutputDataType cell2GateInputWeight;

  //! Weights between the output and forget gate.
  OutputDataType output2GateForgetWeight;

  //! Weights between the input and gate.
  OutputDataType input2GateForgetWeight;

  //! Bias between the input and gate.
  OutputDataType input2GateForgetBias;

  //! Bias between the input and gate.
  OutputDataType cell2GateForgetWeight;

  //! Weights between the output and gate.
  OutputDataType output2GateOutputWeight;

  //! Weights between the input and gate.
  OutputDataType input2GateOutputWeight;

  //! Bias between the input and gate.
  OutputDataType input2GateOutputBias;

  //! Weights between cell and output gate.
  OutputDataType cell2GateOutputWeight;

  //! Locally-stored input gate parameter.
  OutputDataType inputGate;

  //! Locally-stored forget gate parameter.
  OutputDataType forgetGate;

  //! Locally-stored hidden layer parameter.
  OutputDataType hiddenLayer;

  //! Locally-stored output gate parameter.
  OutputDataType outputGate;

  //! Locally-stored input gate activation.
  OutputDataType inputGateActivation;

  //! Locally-stored forget gate activation.
  OutputDataType forgetGateActivation;

  //! Locally-stored output gate activation.
  OutputDataType outputGateActivation;

  //! Locally-stored hidden layer activation.
  OutputDataType hiddenLayerActivation;

  //! Locally-stored input to hidden weight.
  OutputDataType input2HiddenWeight;

  //! Locally-stored input to hidden bias.
  OutputDataType input2HiddenBias;

  //! Locally-stored output to hidden weight.
  OutputDataType output2HiddenWeight;

  //! Locally-stored cell parameter.
  OutputDataType cell;

  //! Locally-stored cell activation error.
  OutputDataType cellActivation;

  //! Locally-stored forget gate error.
  OutputDataType forgetGateError;

  //! Locally-stored output gate error.
  OutputDataType outputGateError;

  //! Locally-stored previous error.
  OutputDataType prevError;

  //! Locally-stored output parameters.
  OutputDataType outParameter;

  //! Locally-stored input cell error parameter.
  OutputDataType inputCellError;

  //! Locally-stored input gate error.
  OutputDataType inputGateError;

  //! Locally-stored hidden layer error.
  OutputDataType hiddenError;

  //! Locally-stored current rho size.
  size_t rhoSize;

  //! Current backpropagate through time steps.
  size_t bpttSteps;
}; // class LSTM

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "lstm_impl.hpp"

#endif
