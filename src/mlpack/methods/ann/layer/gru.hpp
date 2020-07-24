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

#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"

#include "layer_types.hpp"
#include "add_merge.hpp"
#include "sequential.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a gru network layer.
 *
 * This cell can be used in RNN networks.
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
class GRU
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
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& /* error */,
                arma::Mat<eT>& /* gradient */);

  /*
   * Resets the cell to accept a new input. This breaks the BPTT chain starts a
   * new one.
   *
   * @param size The current maximum number of steps through time.
   */
  void ResetCell(const size_t size);

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
  std::vector<LayerTypes<> >& Model() { return network; }

  //! Get the number of input units.
  size_t InSize() const { return inSize; }

  //! Get the number of output units.
  size_t OutSize() const { return outSize; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar);

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
  OutputDataType weights;

  //! Locally-stored input 2 gate module.
  LayerTypes<> input2GateModule;

  //! Locally-stored output 2 gate module.
  LayerTypes<> output2GateModule;

  //! Locally-stored output hidden state 2 gate module.
  LayerTypes<> outputHidden2GateModule;

  //! Locally-stored input gate module.
  LayerTypes<> inputGateModule;

  //! Locally-stored hidden state module.
  LayerTypes<> hiddenStateModule;

  //! Locally-stored forget gate module.
  LayerTypes<> forgetGateModule;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! Locally-stored list of network modules.
  std::vector<LayerTypes<> > network;

  //! Locally-stored number of forward steps.
  size_t forwardStep;

  //! Locally-stored number of backward steps.
  size_t backwardStep;

  //! Locally-stored number of gradient steps.
  size_t gradientStep;

  //! Locally-stored output parameters.
  std::list<arma::mat> outParameter;

  //! Matrix of all zeroes to initialize the output
  arma::mat allZeros;

  //! Iterator pointed to the last output produced by the cell
  std::list<arma::mat>::iterator prevOutput;

  //! Iterator pointed to the last output processed by backward
  std::list<arma::mat>::iterator backIterator;

  //! Iterator pointed to the last output processed by gradient
  std::list<arma::mat>::iterator gradIterator;

  //! Locally-stored previous error.
  arma::mat prevError;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class GRU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "gru_impl.hpp"

#endif
