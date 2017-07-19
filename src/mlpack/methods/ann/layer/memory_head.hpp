/**
 * @file memory_unit.hpp
 * @author Sumedh Ghaisas
 *
 * Definition of the LSTM class, which implements a lstm network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEMORY_HEAD_HPP
#define MLPACK_METHODS_ANN_LAYER_MEMORY_HEAD_HPP

#include <mlpack/prereqs.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"
#include "../activation_functions/softplus_function.hpp"

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
class MemoryHead
{
 public:
  /**
   * Create the LSTM layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  MemoryHead(const size_t inSize, const size_t outSize, const size_t memSize, const size_t shiftSize);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::mat&& memory, arma::Mat<eT>&& output);

  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    Forward(std::move(input), arma::ones<arma::mat>(outSize, memSize), std::move(output));
  }

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
                arma::Mat<eT>&& g);

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
                arma::Mat<eT>&& /* gradient */);

  /*
   * Resets the cell to accept a new input.
   * This breaks the BPTT chain starts a new one.
   */
  void ResetCell();

  //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

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
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Number of steps to backpropagate through time (BPTT).
  size_t memSize;

  size_t shiftSize;

  std::list<arma::mat> prevWeights;
  std::list<arma::mat>::iterator weightsBackwardIterator;

  arma::mat prevError;

  std::list<arma::vec> l_s_t;
  std::list<arma::vec>::iterator b_s_t;

  std::list<arma::mat> l_shiftMatrix;
  std::list<arma::mat>::iterator b_shiftMatrix;

  std::list<arma::mat> l_memory_t;
  std::list<arma::mat>::iterator b_memory_t;

  std::list<arma::vec> l_w_e;
  std::list<arma::vec>::iterator b_w_e;

  std::list<arma::vec> l_w_c;
  std::list<arma::vec>::iterator b_w_c;

  std::list<arma::vec> l_w_g;
  std::list<arma::vec>::iterator b_w_g;

  std::list<arma::vec> l_w_tilde;
  std::list<arma::vec>::iterator b_w_tilde;

  std::list<arma::vec> l_w_dash;
  std::list<arma::vec>::iterator b_w_dash;

  std::list<arma::vec> l_cosine_t;
  std::list<arma::vec>::iterator b_cosine_t;

  arma::vec prev_d_w;

  std::list<double> l_gamma_t;
  std::list<double>::iterator b_gamma_t;

  std::list<double> l_g_t;
  std::list<double>::iterator b_g_t;

  std::list<double> l_b_t;
  std::list<double>::iterator b_b_t;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored input 2 gate module.
  LayerTypes input_linear;

  //! Locally-stored output 2 gate module.
  LayerTypes k_t_non_linear;

  //! Locally-stored input gate module.
  SoftplusFunction b_t_non_linear;

  //! Locally-stored hidden state module.
  LogisticFunction g_t_non_linear;

  //! Locally-stored forget gate module.
  SoftplusFunction gamma_t_non_linear;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored list of network modules.
  std::vector<LayerTypes> network;

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

// Include implementation.
#include "memory_head_impl.hpp"

#endif
