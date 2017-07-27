/**
 * @file memory_unit.hpp
 * @author Sumedh Ghaisas
 *
 * Definition of Memory Head used Neural Turing Machine
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
 * An implementation of a memory head used in NTM.
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
   * Create the Memory Head layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize Size of the output weight vector.
   * @param memSize Memory size in each memory block.
   * @param shiftSize Circular convolutional shift size used.
   */
  MemoryHead(const size_t inSize,
             const size_t outSize,
             const size_t memSize,
             const size_t shiftSize);

  /**
   * Feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward given the current memory content.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   * @param memory Current memory content.
   */
  template<typename eT>
  void ForwardWithMemory(arma::Mat<eT>&& input,
                         const arma::Mat<eT>&& memory,
                         arma::Mat<eT>&& output);

  /**
   * Feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward given the current memory content.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   * @param memory Current memory content.
   */
  template<typename eT>
  void ForwardWithMemoryTest(arma::Mat<eT>&& input,
                             const arma::Mat<eT>&& memory,
                             arma::Mat<eT>&& output)
  {
    ForwardWithMemory(std::move(input),
                      std::move(memory),
                      std::move(output));
  }

  /**
   * Feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward.
   *
   * Function used for testing the class. Creates fixed memory to be used in
   * forward propagation.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   * @param memory Current memory content.
   */
  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    arma::mat memory = arma::ones<arma::mat>(outSize, memSize);
    ForwardWithMemory(std::move(input), std::move(memory), std::move(output));
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param memory The current memory content.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void BackwardWithMemory(const arma::Mat<eT>&& /* output */,
                          const arma::Mat<eT>&& /* input */,
                          const arma::Mat<eT>&& memory,
                          arma::Mat<eT>&& gy,
                          arma::Mat<eT>&& g,
                          arma::Mat<eT>&& gM);

  template<typename eT>
  void Backward(const arma::Mat<eT>&& output,
                const arma::Mat<eT>&& input,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g)
  {
    arma::mat dM;
    arma::mat memory = arma::ones<arma::mat>(outSize, memSize);
    BackwardWithMemory(std::move(output), std::move(input), std::move(memory),
        std::move(gy), std::move(g), std::move(dM));
  }

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

  //! Memory size in each memory block.
  size_t memSize;

  //! Shifting size used in circular convolution.
  size_t shiftSize;

  //! Store the previous output weights.
  std::list<arma::mat> prevWeights;

  //! Iterator to previous output weights ,used by backward.
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

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored input 2 gate module.
  LayerTypes inputLinear;

  //! Locally-stored output 2 gate module.
  LayerTypes kTNonLinear;

  //! Locally-stored input gate module.
  SoftplusFunction bTNonLinear;

  //! Locally-stored hidden state module.
  SoftplusFunction gTNonLinear;

  //! Locally-stored forget gate module.
  SoftplusFunction gammaTNonLinear;

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
