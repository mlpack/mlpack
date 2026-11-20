// Temporarily drop.
/**
 * @file methods/ann/layer/fast_lstm.hpp
 * @author Marcus Edel
 *
 * Definition of the Fast LSTM class, which implements a Fast LSTM network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_FAST_LSTM_HPP
#define MLPACK_METHODS_ANN_LAYER_FAST_LSTM_HPP

#include <mlpack/prereqs.hpp>
#include <limits>
#include "layer.hpp"

namespace mlpack {

/**
 * An implementation of a faster version of the Fast LSTM network layer.
 * Basically by combining the calculation of the input, forget, output gates
 * and hidden state in a single step. The standard formula changes as follows:
 *
 * @f{eqnarray}{
 * i &=& sigmoid(W \cdot x + W \cdot h + b) \\
 * f &=& sigmoid(W  \cdot x + W \cdot h + b) \\
 * z &=& tanh(W \cdot x + W \cdot h + b) \\
 * c &=& f \cdot c + i \cdot z \\
 * o &=& sigmoid(W \cdot x + W \cdot h + b) \\
 * h &=& o \cdot tanh(c)
 * @f}
 *
 * Note that FastLSTM network layer does not use peephole connections between
 * the cell and gates.
 *
 * Note also that if a FastLSTM layer is desired as the first layer of a neural
 * network, an IdentityLayer should be added to the network as the first layer,
 * and then the FastLSTM layer should be added.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Hochreiter1997,
 *   author  = {Hochreiter, Sepp and Schmidhuber, J\"{u}rgen},
 *   title   = {Long Short-term Memory},
 *   journal = {Neural Comput.},
 *   year    = {1997},
 *   url     = {https://www.bioinf.jku.at/publications/older/2604.pdf}
 * }
 * @endcode
 *
 * \see LSTM for a standard implementation of the LSTM layer.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class FastLSTMType : public Layer<InputType, OutputType>
{
 public:
  // Convenience typedefs.
  using InputET = typename InputType::elem_type;
  using OutputET = typename OutputType::elem_type;

  //! Create the FastLSTMType object.
  FastLSTMType();

  //! Copy Constructor
  FastLSTMType(const FastLSTMType& layer);

  //! Move Constructor
  FastLSTMType(FastLSTMType&& layer);

  //! Copy assignment operator
  FastLSTMType& operator=(const FastLSTMType& layer);

  //! Move assignment operator
  FastLSTMType& operator=(FastLSTMType&& layer);

  /**
   * Create the Fast LSTM layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  FastLSTMType(const size_t inSize,
               const size_t outSize,
               const size_t rho = std::numeric_limits<size_t>::max());

  //! Clone the FastLSTMType object. This handles polymorphism correctly.
  FastLSTMType* Clone() const { return new FastLSTMType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input,
                const OutputType& gy,
                OutputType& g);

  /**
   * Reset the layer parameter.
   */
  void Reset();

  /**
   * Resets the cell to accept a new input. This breaks the BPTT chain starts a
   * new one.
   *
   * @param size The current maximum number of steps through time.
   */
  void ResetCell(const size_t size);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the maximum number of steps to backpropagate through time (BPTT).
  size_t Rho() const { return rho; }
  //! Modify the maximum number of steps to backpropagate through time (BPTT).
  size_t& Rho() { return rho; }

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the number of input units.
  size_t InSize() const { return inSize; }

  //! Get the number of output units.
  size_t OutSize() const { return outSize; }

  //! Get the size of the weight matrix.
  size_t WeightSize() const
  {
    return 4 * outSize * inSize + 4 * outSize + 4 * outSize * outSize;
  }

  const std::vector<size_t> OutputDimensions() const
  {
    std::vector<size_t> result(inputDimensions.size(), 0);
    result[0] = outSize;
    return result;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * This speeds up the sigmoid operation by using an approximation.
   *
   * @param input The input data.
   * @param sigmoid The matrix to store the sigmoid approximation into.
   */
  void FastSigmoid(const InputType& input, OutputType& sigmoids)
  {
    for (size_t i = 0; i < input.n_elem; ++i)
      sigmoids(i) = FastSigmoid(input(i));
  }

  /**
   * Sigmoid approximation for the given sample.
   *
   * @param data The given data sample for the sigmoid approximation.
   * @tparam The sigmoid approximation.
   */
  OutputET FastSigmoid(const InputET data)
  {
    OutputET x = 0.5 * data;
    OutputET z;
    if (x >= 0)
    {
      if (x < 1.7)
        z = (1.5 * x / (1 + x));
      else if (x < 3)
        z = (0.935409070603099 + 0.0458812946797165 * (x - 1.7));
      else
        z = 0.99505475368673;
    }
    else
    {
      OutputET xx = -x;
      if (xx < 1.7)
        z = -(1.5 * xx / (1 + xx));
      else if (xx < 3)
        z = -(0.935409070603099 + 0.0458812946797165 * (xx - 1.7));
      else
        z = -0.99505475368673;
    }

    return 0.5 * (z + 1.0);
  }

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
  OutputType weights;

  //! Locally-stored previous output.
  OutputType prevOutput;

  //! Locally-stored batch size.
  size_t batchSize;

  //! Current batch step, alias for batchSize - 1.
  size_t batchStep;

  //! Current gradient step to keep track of the backpropagate through time
  //! step.
  size_t gradientStepIdx;

  //! Locally-stored cell activation error.
  OutputType cellActivationError;

  //! Locally-stored output parameter object.
  OutputType outputParameter;

  //! Weights between the output and gate.
  OutputType output2GateWeight;

  //! Weights between the input and gate.
  OutputType input2GateWeight;

  //! Bias between the input and gate.
  OutputType input2GateBias;

  //! Locally-stored gate parameter.
  OutputType gate;

  //! Locally-stored gate activation.
  OutputType gateActivation;

  //! Locally-stored state activation.
  OutputType stateActivation;

  //! Locally-stored cell parameter.
  OutputType cell;

  //! Locally-stored cell activation error.
  OutputType cellActivation;

  //! Locally-stored foget gate error.
  OutputType forgetGateError;

  //! Locally-stored previous error.
  OutputType prevError;

  //! Locally-stored current rho size.
  size_t rhoSize;

  //! Current backpropagate through time steps.
  size_t bpttSteps;
}; // class FastLSTMType.

// Standard FastLSTM layer.
using FastLSTM = FastLSTMType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "fast_lstm_impl.hpp"

#endif
