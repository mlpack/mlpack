/**
 * @file fast_lstm.hpp
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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

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
 *   year    = {1997}
 * }
 * @endcode
 *
 * \see LSTM for a standard implementation of the LSTM layer.
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
class FastLSTM
{
 public:
  // Convenience typedefs.
  typedef typename InputDataType::elem_type InputElemType;
  typedef typename OutputDataType::elem_type ElemType;

  //! Create the Fast LSTM object.
  FastLSTM();

  /**
   * Create the Fast LSTM layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  FastLSTM(const size_t inSize,
           const size_t outSize,
           const size_t rho = std::numeric_limits<size_t>::max());

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(InputType&& input, OutputType&& output);

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
  /**
   * This speeds up the sigmoid operation by using an approximation.
   *
   * @param input The input data.
   * @param sigmoid The matrix to store the sigmoid approximation into.
   */
  template<typename InputType, typename OutputType>
  void FastSigmoid(InputType&& input, OutputType&& sigmoids)
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
  ElemType FastSigmoid(const InputElemType data)
  {
    ElemType x = 0.5 * data;
    ElemType z;
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
      ElemType xx = -x;
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

  //! Weights between the output and gate.
  OutputDataType output2GateWeight;

  //! Weights between the input and gate.
  OutputDataType input2GateWeight;

  //! Bias between the input and gate.
  OutputDataType input2GateBias;

  //! Locally-stored gate parameter.
  OutputDataType gate;

  //! Locally-stored gate activation.
  OutputDataType gateActivation;

  //! Locally-stored state activation.
  OutputDataType stateActivation;

  //! Locally-stored cell parameter.
  OutputDataType cell;

  //! Locally-stored cell activation error.
  OutputDataType cellActivation;

  //! Locally-stored foget gate error.
  OutputDataType forgetGateError;

  //! Locally-stored previous error.
  OutputDataType prevError;

  //! Locally-stored output parameters.
  OutputDataType outParameter;

  //! Locally-stored current rho size.
  size_t rhoSize;

  //! Current backpropagate through time steps.
  size_t bpttSteps;
}; // class FastLSTM

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "fast_lstm_impl.hpp"

#endif
