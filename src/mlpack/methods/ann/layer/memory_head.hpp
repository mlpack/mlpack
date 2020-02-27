/**
 * @file memory_unit.hpp
 * @author Sumedh Ghaisas
 *
 * Definition of Memory Head layer used in Neural Turing Machine.
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
 * Given an input, MemoryHead computes weights over which the memory locations
 * will be accessed. Each memory location will be given a weight and the sum of
 * weights is always one.
 *
 * MemoryHead is used in Neural Turing Machine to compute the weights for
 * reading and writing memory.
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
  //! Create the MemoryHead object.
  MemoryHead();

  /**
   * Create the Memory Head layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize Size of the output weight vector.
   * @param memSize Memory size in each memory block.
   * @param shiftSize Circular convolutional shift size used.
   * @param memoryHistory Access to the memory
   * @param dMem Location to store the gradient w.r.t. memory
   */
  MemoryHead(const size_t inSize,
             const size_t outSize,
             const size_t memSize,
             const size_t shiftSize,
             const std::list<arma::mat>& memoryHistory,
             arma::mat& dMem);

  /**
   * Delete the MemoryHead and the layers it holds.
   */
  ~MemoryHead();

  /**
   * Feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward given the current memory content.
   * Current memory content is taken from memoryHistory.
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
   * This function is used in testing the layer without MemoryTest layer.
   *
   * @param input The propagated input activation.
   * @param memory The current memory content.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename ErrorType, typename GradientType>
  void Backward(const InputType&& input,
                ErrorType&& gy,
                GradientType&& g);

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

  /*
   * Resets the cell to accept a new input.
   * This breaks the BPTT chain starts a new one.
   */
  void ResetCell(const size_t size);

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
  OutputDataType const& Gradient() const { return grad; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return grad; }

  //! Get the model modules.
  std::vector<LayerTypes<>>& Model() { return network; }

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

  //! Memory size in each memory block.
  size_t memSize;

  //! Shifting size used in circular convolution.
  size_t shiftSize;

  //! All zeros vector for initialization.
  arma::mat allZeros;

  //! Forward pass count.
  size_t forwardStep;

  //! Store the previous output weights.
  std::list<arma::mat> prevWeights;

  //! Iterator to previous output weights, used by backward.
  std::list<arma::mat>::iterator weightsBackwardIterator;

  //! Store the delta received at the linear input layer.
  arma::mat prevError;

  //! Store the computed St gate values.
  std::list<arma::vec> lSt;

  //! Iterator to St gate values, used by backward.
  std::list<arma::vec>::iterator bSt;

  //! Store the generated shift matrices.
  std::list<arma::mat> lShiftMatrix;

  //! Iterator to shift matrices, used by backward.
  std::list<arma::mat>::iterator bShiftMatrix;

  //! Store We gate values.
  std::list<arma::vec> lWe;

  //! Iterator to We gate values, used by backward.
  std::list<arma::vec>::iterator bWe;

  //! Store Wc gate values.
  std::list<arma::vec> lWc;

  //! Iterator to Wc gate values, used by backward.
  std::list<arma::vec>::iterator bWc;

  //! Store Wg gate values.
  std::list<arma::vec> lWg;

  //! Iterator to Wg gate values, used by backward.
  std::list<arma::vec>::iterator bWg;

  //! Store W_tilde gate values.
  std::list<arma::vec> lWTilde;

  //! Iterator to W_tilde gate values, used by backward.
  std::list<arma::vec>::iterator bWTilde;

  //! Store W_dash gate values.
  std::list<arma::vec> lWDash;

  //! Iterator to W_dash gate values, used by backward.
  std::list<arma::vec>::iterator bWdash;

  //! Store cosine similarity values.
  std::list<arma::vec> lConsineT;

  //! Iterator to cosine similarity values, used by backward.
  std::list<arma::vec>::iterator bCosineT;

  //! Store gamma_t gate values.
  std::list<double> lGammaT;

  //! Iterator to gamma_t gate values, used by backward.
  std::list<double>::iterator bGammaT;

  //! Store gt gate values.
  std::list<double> lGt;

  //! Iterator to gt gate values, used by backward.
  std::list<double>::iterator bGt;

  //! Store bt gate values.
  std::list<double> lBt;

  //! Iterator to bt gate values, used by backward.
  std::list<double>::iterator bBt;

  //! Store the delta received with BPTT.
  arma::vec prevDW;

  //! Dummy memoryHistory for deafult construction.
  std::list<arma::mat> dummyMemoryHistory;

  //! Access to the memory history.
  const std::list<arma::mat>& memoryHistory;

  //! Backward memory history iterator.
  std::list<arma::mat>::const_iterator bMemoryHistory;

  //! Dummy dMem for default construction.
  arma::mat dummyDMem;

  //! Reference to memory gradient.
  arma::mat& dMem;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored input 2 gate module.
  LayerTypes<> inputLinear;

  //! Locally-stored output 2 gate module.
  LayerTypes<> kTNonLinear;

  //! Locally-stored input gate module.
  SoftplusFunction bTNonLinear;

  //! Locally-stored hidden state module.
  LogisticFunction gTNonLinear;

  //! Locally-stored forget gate module.
  SoftplusFunction gammaTNonLinear;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! Locally-stored list of network modules.
  std::vector<LayerTypes<>> network;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType grad;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class MemoryHead

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "memory_head_impl.hpp"

#endif
