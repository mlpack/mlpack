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
 * Implementation of the "vanilla" LSTM with peephole connections.
 * The implementation corresponds to the following algorithm:
 *
 * z_t  =    tanh(W_z x_t + R_z y_{t - 1} + b_z)
 * i_t  = sigmoid(W_i x_t + R_i y_{t - 1} + p_i % c_{t - 1} + b_i)
 * f_t  = sigmoid(W_f x_t + R_f y_{t - 1} + p_f % c_{t - 1} + b_f)
 * c_t  =         z_t % i_t + c_{t - 1} % f_t
 * o_t  = sigmoid(W_o x_t + R_o y_{t - 1} + p_o % c_t + b_o)
 * y_t  =    tanh(c_t) % o_t
 *
 * which is the 'vanilla' implementation described in the following paper:
 *
 * ```
 * @article{greff2016lstm,
 *   title={LSTM: A search space odyssey},
 *   author={Greff, Klaus and Srivastava, Rupesh K and Koutn{\'\i}k, Jan and
 *       Steunebrink, Bas R and Schmidhuber, J{\"u}rgen},
 *   journal={IEEE transactions on neural networks and learning systems},
 *   volume={28},
 *   number={10},
 *   pages={2222--2232},
 *   year={2016},
 *   publisher={IEEE}
 * }
 * ```
 *
 * See `FastLSTM` for a faster LSTM version which combines the calculation of
 * the input, forget, output gates and hidden state in a single step.
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
  void SetWeights(const MatType& weightsIn);

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

  // Get the parameters.
  const MatType& Parameters() const { return weights; }
  // Modify the parameters.
  MatType& Parameters() { return weights; }

  // Get the block input weight matrix for non-recurrent input.
  const MatType& BlockInputWeight() const { return blockInputWeight; }
  // Modify the block input weight matrix for non-recurrent input.
  MatType& BlockInputWeight() { return blockInputWeight; }
  // Get the input gate weight matrix for non-recurrent input.
  const MatType& InputGateWeight() const { return inputGateWeight; }
  // Modify the input gate weight matrix for non-recurrent input.
  MatType& InputGateWeight() { return inputGateWeight; }
  // Get the forget gate weight matrix for non-recurrent input.
  const MatType& ForgetGateWeight() const { return forgetGateWeight; }
  // Modify the forget gate weight matrix for non-recurrent input.
  MatType& ForgetGateWeight() { return forgetGateWeight; }
  // Get the output gate weight matrix for non-recurrent input.
  const MatType& OutputGateWeight() const { return outputGateWeight; }
  // Modify the output gate weight matrix for non-recurrent input.
  MatType& OutputGateWeight() { return outputGateWeight; }

  // Get the bias vector that is added to the block input.
  const MatType& BlockInputBias() const { return blockInputBias; }
  // Modify the bias vector that is added to the block input.
  MatType& BlockInputBias() { return blockInputBias; }
  // Get the bias vector that is added to the input gate.
  const MatType& InputGateBias() const { return inputGateBias; }
  // Modify the bias vector that is added to the input gate.
  MatType& InputGateBias() { return inputGateBias; }
  // Get the bias vector that is added to the forget gate.
  const MatType& ForgetGateBias() const { return forgetGateBias; }
  // Modify the bias vector that is added to the forget gate.
  MatType& ForgetGateBias() { return forgetGateBias; }
  // Get the bias vector that is added to the output gate.
  const MatType& OutputGateBias() const { return outputGateBias; }
  // Modify the bias vector that is added to the output gate.
  MatType& OutputGateBias() { return outputGateBias; }

  // Get the block input weight matrix for recurrent input.
  const MatType& RecurrentBlockInputWeight() const
  { return recurrentBlockInputWeight; }
  // Modify the block input weight matrix for recurrent input.
  MatType& RecurrentBlockInputWeight() { return recurrentBlockInputWeight; }
  // Get the input gate weight matrix for recurrent input.
  const MatType& RecurrentInputGateWeight() const
  { return recurrentInputGateWeight; }
  // Modify the input gate weight matrix for recurrent input.
  MatType& RecurrentInputGateWeight() { return recurrentInputGateWeight; }
  // Get the forget gate weight matrix for recurrent input.
  const MatType& RecurrentForgetGateWeight() const
  { return recurrentForgetGateWeight; }
  // Modify the forget gate weight matrix for recurrent input.
  MatType& RecurrentForgetGateWeight() { return recurrentForgetGateWeight; }
  // Get the output gate weight matrix for recurrent input.
  const MatType& RecurrentOutputGateWeight() const
  { return recurrentOutputGateWeight; }
  // Modify the output gate weight matrix for recurrent input.
  MatType& RecurrentOutputGateWeight() { return recurrentOutputGateWeight; }

  // Get the peephole input gate weight matrix.
  const MatType& PeepholeInputGateWeight() const
  { return peepholeInputGateWeight; }
  // Modify the peephole input gate weight matrix.
  MatType& PeepholeInputGateWeight() { return peepholeInputGateWeight; }
  // Get the peephole forget gate weight matrix.
  const MatType& PeepholeForgetGateWeight() const
  { return peepholeForgetGateWeight; }
  // Modify the peephole forget gate weight matrix.
  MatType& PeepholeForgetGateWeight() { return peepholeForgetGateWeight; }
  // Get the peephole output gate weight matrix.
  const MatType& PeepholeOutputGateWeight() const
  { return peepholeOutputGateWeight; }
  // Modify the peephole output gate weight matrix.
  MatType& PeepholeOutputGateWeight() { return peepholeOutputGateWeight; }

  // Get the total number of trainable parameters.
  size_t WeightSize() const;

  // Get the total number of recurrent state parameters.
  size_t RecurrentSize() const;

  // Given a properly set InputDimensions(), compute the output dimensions.
  void ComputeOutputDimensions()
  {
    inSize = this->inputDimensions[0];
    for (size_t i = 1; i < this->inputDimensions.size(); ++i)
      inSize *= this->inputDimensions[i];
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
  // Locally-stored number of input units.
  size_t inSize;

  // Locally-stored number of output units.
  size_t outSize;

  // Locally-stored weight object.
  MatType weights;

  // Weight matrices for each of the input connections.
  MatType blockInputWeight;
  MatType inputGateWeight;
  MatType forgetGateWeight;
  MatType outputGateWeight;

  // Bias vectors for each of the input connections.
  MatType blockInputBias;
  MatType inputGateBias;
  MatType forgetGateBias;
  MatType outputGateBias;

  // Weight matrices for each of the recurrent connections.
  MatType recurrentBlockInputWeight;
  MatType recurrentInputGateWeight;
  MatType recurrentForgetGateWeight;
  MatType recurrentOutputGateWeight;

  // Peephole weight connections for each of the recurrent cell connections.
  MatType peepholeInputGateWeight;
  MatType peepholeForgetGateWeight;
  MatType peepholeOutputGateWeight;

  // These matrices are internally used for computation only; they are aliases
  // for recurrent state.
  MatType blockInput;
  MatType inputGate;
  MatType forgetGate;
  MatType outputGate;
  MatType thisRecurrent;
  MatType prevRecurrent;
  MatType thisCell;
  MatType prevCell;
  MatType thisY;

  // These matrices are also internally used for computation only.
  // Everything below 'workspace' is an alias of memory in 'workspace'.
  //
  // TODO: right now these are stored inside the class itself, but that is not a
  // great design; we need some notion of 'working space' that a layer can have.
  MatType workspace;
  MatType deltaY;
  MatType deltaBlockInput;
  MatType deltaInputGate;
  MatType deltaForgetGate;
  MatType deltaOutputGate;
  MatType deltaCell;
  // These correspond to, e.g., dy_{t + 1}.
  MatType nextDeltaY;
  MatType nextDeltaBlockInput;
  MatType nextDeltaInputGate;
  MatType nextDeltaForgetGate;
  MatType nextDeltaOutputGate;
  MatType nextDeltaCell;

  // Calling this function will set all the aliases for the functions above to
  // the correct places in the current recurrent state methods.
  void SetInternalAliases(const size_t batchSize);

  // Calling this function will set up workspace memory for the backward pass,
  // if necessary.
  void SetBackwardWorkspace(const size_t batchSize);
}; // class LSTMType

// Convenience typedefs.

// Standard LSTM layer.
using LSTM = LSTMType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "lstm_impl.hpp"

#endif
