/**
 * @file neural_turing_machine.hpp
 * @author Sumedh Ghaisas
 *
 * Definition of the Neural Turing Machine class.
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{graves2014neural,
 *    title   = {Neural turing machines},
 *    author  = {Graves, Alex and Wayne, Greg and Danihelka, Ivo},
 *    journal = {arXiv preprint arXiv:1410.5401},
 *    year    = {2014}
 * }
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NTM_HPP
#define MLPACK_METHODS_ANN_LAYER_NTM_HPP

#include <mlpack/prereqs.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"
#include "../activation_functions/softplus_function.hpp"

#include "layer_types.hpp"
#include "add_merge.hpp"
#include "sequential.hpp"

#include "memory_head.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * In Neural Turing Mahine, the expressiveness of a neural network,
 * (or controller) is increased by attaching a memory to it. The input to this
 * modified neural network consists of original input and the read memory. In
 * turn the controller decides which memory locations to read and which memory
 * locations to write. The controller output is taken as the final output.
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
class NeuralTuringMachine
{
 public:
  //! Create the NeuralTuringMachine object.
  NeuralTuringMachine();


  /**
   * Create the Neural Turing Machine layer object using the specified
   * parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param numMem Number of memory locations to use.
   * @param memSize Size of each memory location.
   * @param shiftSize Circular shift rotation size.
   * @param controller The controller network to use.
   */
  NeuralTuringMachine(const size_t inSize,
                      const size_t outSize,
                      const size_t numMem,
                      const size_t memSize,
                      const size_t shiftSize,
                      LayerTypes<> controller);

  /**
   * Delete the NeuralTuringMachine and the layers it holds.
   */
  ~NeuralTuringMachine();

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

  //! Number of memory locations.
  size_t numMem;

  //! Size of memory locations.
  size_t memSize;

  //! Circular shift rotation size.
  size_t shiftSize;

  //! Memory read head.
  LayerTypes<> readHead;

  //! Locally stored read head error.
  arma::mat dReadHead;

  //! Locally stored memory write error.
  arma::mat dWriteHead;

  //! Linear layer to generate erase and add vectors.
  LayerTypes<> inputToLinear;

  //! Non linearity for Add operation.
  LayerTypes<> addGate;

  //! Non linearity for Erase operation.
  LayerTypes<> eraseGate;

  //! Memory head to generate write weights.
  LayerTypes<> writeHead;

  //! Locally stored error for linear layer.
  arma::mat linearError;

  //! Controller.
  LayerTypes<> controller;

  //! Locally stored memory content error.
  arma::mat dMem;

  //! All zeros vector for initializing read.
  arma::mat allZeros;

  //! All ones matrix for initializing memory.
  arma::mat allOnes;

  //! Storing all the memory contents for backward pass.
  std::list<arma::mat> memoryHistory;

  //! Backward pass iterator to stored memory content.
  std::list<arma::mat>::iterator bMemoryHistory;

  //! Storing all the memory reads for backward pass.
  std::list<arma::mat> lReads;

  //! Backward pass iterator to stored memory reads.
  std::list<arma::mat>::iterator gReads;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! Locally-stored list of network modules.
  std::vector<LayerTypes<>> network;

  //! Locally-stored number of forward steps.
  size_t forwardStep;

  //! Locally-stored number of backward steps.
  size_t backwardStep;

  //! Locally-stored number of gradient steps.
  size_t gradientStep;

  //! Locally-stored previous error.
  arma::mat prevError;

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
}; // class NeuralTuringMachine

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "neural_turing_machine_impl.hpp"

#endif
