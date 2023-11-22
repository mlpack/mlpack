/**
 * @file methods/ann/layer/lstm.hpp
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

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the LSTM module class.
 * The implementation corresponds to the following algorithm:
 *
 * @f{eqnarray}{
 * i &=& sigmoid(W \cdot x + W \cdot h + W \cdot c + b) \\
 * f &=& sigmoid(W  \cdot x + W \cdot h + W \cdot c + b) \\
 * z &=& tanh(W \cdot x + W \cdot h + b) \\
 * c &=& f \odot c + i \odot z \\
 * o &=& sigmoid(W \cdot x + W \cdot h + W \cdot c + b) \\
 * h &=& o \odot tanh(c)
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
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class LSTMType : public RecurrentLayer<MatType>
{
 public:
  //! Create the LSTM object.
  LSTMType();

  /**
   * Create the LSTM layer object using the specified parameters.
   *
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  LSTMType(const size_t outSize);

  //! Clone the LSTMType object. This handles polymorphism correctly.
  LSTMType* Clone() const { return new LSTMType(*this); }

  //! Copy the given LSTMType object.
  LSTMType(const LSTMType& other);
  //! Take ownership of the given LSTMType object's data.
  LSTMType(LSTMType&& other);
  //! Copy the given LSTMType object.
  LSTMType& operator=(const LSTMType& other);
  //! Take ownership of the given LSTMType object's data.
  LSTMType& operator=(LSTMType&& other);

  virtual ~LSTMType() { }

  /**
   * Reset the layer parameter. The method is called to
   * assign the allocated memory to the internal learnable parameters.
   */
  void SetWeights(typename MatType::elem_type* weightsPtr);

  /**
   * Ordinary feed-forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient);

  /**
   * Reset the recurrent state of the LSTM layer, and allocate enough space to
   * hold `bpttSteps` of previous passes with a batch size of `batchSize`.
   *
   * @param bpttSteps Number of steps of history to allocate space for.
   * @param batchSize Batch size to prepare for.
   */
  void ClearRecurrentState(const size_t bpttSteps, const size_t batchSize);

  //! Get the parameters.
  const MatType& Parameters() const { return weights; }
  //! Modify the parameters.
  MatType& Parameters() { return weights; }

  //! Get the total number of trainable parameters.
  size_t WeightSize() const
  {
    return (4 * outSize * inSize + 7 * outSize + 4 * outSize * outSize);
  }

  //! Given a properly set InputDimensions(), compute the output dimensions.
  void ComputeOutputDimensions()
  {
    inSize = std::accumulate(this->inputDimensions.begin(),
        this->inputDimensions.end(), 0);
    this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
        1);

    // The LSTM layer flattens its input.
    this->outputDimensions[0] = outSize;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  MatType weights;

  //! Weights between the output and input gate.
  MatType output2GateInputWeight;

  //! Weights between the input and gate.
  MatType input2GateInputWeight;

  //! Bias between the input and input gate.
  MatType input2GateInputBias;

  //! Weights between the cell and input gate.
  MatType cell2GateInputWeight;

  //! Weights between the output and forget gate.
  MatType output2GateForgetWeight;

  //! Weights between the input and gate.
  MatType input2GateForgetWeight;

  //! Bias between the input and gate.
  MatType input2GateForgetBias;

  //! Bias between the input and gate.
  MatType cell2GateForgetWeight;

  //! Weights between the output and gate.
  MatType output2GateOutputWeight;

  //! Weights between the input and gate.
  MatType input2GateOutputWeight;

  //! Bias between the input and gate.
  MatType input2GateOutputBias;

  //! Weights between cell and output gate.
  MatType cell2GateOutputWeight;

  // Below here are recurrent state matrices.

  //! Locally-stored input gate parameter.
  MatType inputGate;

  //! Locally-stored forget gate parameter.
  MatType forgetGate;

  //! Locally-stored hidden layer parameter.
  MatType hiddenLayer;

  //! Locally-stored output gate parameter.
  MatType outputGate;

  //! Locally-stored input to hidden weight.
  MatType input2HiddenWeight;

  //! Locally-stored input to hidden bias.
  MatType input2HiddenBias;

  //! Locally-stored output to hidden weight.
  MatType output2HiddenWeight;

  //! Locally-stored cell parameter.
  arma::Cube<typename MatType::elem_type> cell;

  // These members store recurrent state.

  //! Locally-stored input gate activation.
  arma::Cube<typename MatType::elem_type> inputGateActivation;

  //! Locally-stored forget gate activation.
  arma::Cube<typename MatType::elem_type> forgetGateActivation;

  //! Locally-stored output gate activation.
  arma::Cube<typename MatType::elem_type> outputGateActivation;

  //! Locally-stored hidden layer activation.
  arma::Cube<typename MatType::elem_type> hiddenLayerActivation;

  //! Locally-stored cell activation error.
  arma::Cube<typename MatType::elem_type> cellActivation;

  //! Locally-stored forget gate error.
  MatType forgetGateError;

  //! Locally-stored output gate error.
  MatType outputGateError;

  //! Locally-stored output parameters.
  arma::Cube<typename MatType::elem_type> outParameter;

  //! Locally-stored input cell error parameter.
  MatType inputCellError;

  //! Locally-stored input gate error.
  MatType inputGateError;

  //! Locally-stored hidden layer error.
  MatType hiddenError;
}; // class LSTMType

// Convenience typedefs.

// Standard LSTM layer.
typedef LSTMType<arma::mat> LSTM;

} // namespace mlpack

// Include implementation.
#include "lstm_impl.hpp"

#endif
