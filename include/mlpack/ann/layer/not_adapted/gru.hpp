// Temporarily drop.
/**
 * @file methods/ann/layer/gru.hpp
 * @author Sumedh Ghaisas
 *
 * Definition of the GRU layer.
 *
 * For more information, read the following paper:
 *
 * @code
 * @inproceedings{chung2015gated,
 *    title     = {Gated Feedback Recurrent Neural Networks.},
 *    author    = {Chung, Junyoung and G{\"u}l{\c{c}}ehre, Caglar and Cho,
                  Kyunghyun and Bengio, Yoshua},
 *    booktitle = {ICML},
 *    pages     = {2067--2075},
 *    year      = {2015},
 *    url       = {https://arxiv.org/abs/1502.02367}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GRU_HPP
#define MLPACK_METHODS_ANN_LAYER_GRU_HPP

#include <list>
#include <limits>

#include <mlpack/prereqs.hpp>

#include "layer_types.hpp"
#include "add_merge.hpp"
#include "sequential.hpp"

namespace mlpack {

/**
 * An implementation of a gru network layer.
 *
 * This cell can be used in RNN networks.
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
class GRU : public Layer<InputType, OutputType>
{
 public:
  //! Create the GRU object.
  GRU();

  /**
   * Create the GRU layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  GRU(const size_t inSize,
      const size_t outSize,
      const size_t rho = std::numeric_limits<size_t>::max());

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
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& /* error */,
                OutputType& /* gradient */);

  /*
   * Resets the cell to accept a new input. This breaks the BPTT chain starts a
   * new one.
   *
   * @param size The current maximum number of steps through time.
   */
  void ResetCell(const size_t size);

  //! Get the maximum number of steps to backpropagate through time (BPTT).
  size_t Rho() const { return rho; }
  //! Modify the maximum number of steps to backpropagate through time (BPTT).
  size_t& Rho() { return rho; }

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the model modules.
  std::vector<Layer<InputType, OutputType>*>& Model() { return network; }

  //! Get the number of input units.
  size_t InSize() const { return inSize; }

  //! Get the number of output units.
  size_t OutSize() const { return outSize; }

  //! Get the shape of the input.
  size_t InputShape() const
  {
    return inSize;
  }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Number of steps to backpropagate through time (BPTT).
  size_t rho;

  //! Current batch size.
  size_t batchSize;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored input 2 gate module.
  Layer<InputType, OutputType>* input2GateModule;

  //! Locally-stored output 2 gate module.
  Layer<InputType, OutputType>* output2GateModule;

  //! Locally-stored output hidden state 2 gate module.
  Layer<InputType, OutputType>* outputHidden2GateModule;

  //! Locally-stored input gate module.
  Layer<InputType, OutputType>* inputGateModule;

  //! Locally-stored hidden state module.
  Layer<InputType, OutputType>* hiddenStateModule;

  //! Locally-stored forget gate module.
  Layer<InputType, OutputType>* forgetGateModule;

  //! Locally-stored list of network modules.
  std::vector<Layer<InputType, OutputType>*> network;

  //! Locally-stored number of forward steps.
  size_t forwardStep;

  //! Locally-stored number of backward steps.
  size_t backwardStep;

  //! Locally-stored number of gradient steps.
  size_t gradientStep;

  //! Locally-stored output parameters.
  std::list<OutputType> outParameter;

  //! Matrix of all zeroes to initialize the output
  OutputType allZeros;

  //! Iterator pointed to the last output produced by the cell
  typename std::list<OutputType>::iterator prevOutput;

  //! Iterator pointed to the last output processed by backward
  typename std::list<OutputType>::iterator backIterator;

  //! Iterator pointed to the last output processed by gradient
  typename std::list<OutputType>::iterator gradIterator;

  //! Locally-stored previous error.
  OutputType prevError;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored gradient object.
  OutputType gradient;

  //! Locally-stored output parameter object.
  OutputType outputParameter;
}; // class GRU

} // namespace mlpack

// Include implementation.
#include "gru_impl.hpp"

#endif
