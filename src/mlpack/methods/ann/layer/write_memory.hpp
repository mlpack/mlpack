/**
 * @file write_memory.hpp
 * @author Sumedh Ghaisas
 *
 * Definition of Write Memory used Neural Turing Machine
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_WRITE_MEMORY_HPP
#define MLPACK_METHODS_ANN_LAYER_WRITE_MEMORY_HPP

#include <mlpack/prereqs.hpp>

#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"

#include "layer_types.hpp"

#include "memory_head.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a WriteMemory layer used in NTM.
 * The layer takes the controller output as input and performs a write operation
 * on the given memory content.
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
class WriteMemory
{
 public:
  /**
   * Create the Write Memory layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param numMem Number of memory locations.
   * @param memSize Memory size in each memory block.
   * @param shiftSize Circular convolutional shift size used.
   */
  WriteMemory(const size_t inSize,
              const size_t numMem,
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
   * This function is used in testing the class with MemoryTest layer.
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
    ForwardWithMemory(std::move(input), std::move(memory), std::move(output));

    output = arma::mat(output.memptr(), output.n_rows * output.n_cols, 1);
  }

  /**
   * Feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward.
   *
   * This function is used in testing the layer without MemoryTest layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   * @param memory Current memory content.
   */
  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    arma::mat memory = arma::ones<arma::mat>(numMem, memSize);
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
                          const arma::Mat<eT>&& memory,
                          arma::Mat<eT>&& gy,
                          arma::Mat<eT>&& g,
                          arma::Mat<eT>&& gM);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * This function is used in testing the layer with MemoryTest layer.
   *
   * @param input The propagated input activation.
   * @param memory The current memory content.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void BackwardWithMemoryTest(const arma::Mat<eT>&& output,
                              const arma::Mat<eT>&& memory,
                              arma::Mat<eT>&& gy,
                              arma::Mat<eT>&& g,
                              arma::Mat<eT>&& gM)
  {
    gy = arma::mat(gy.memptr(), memory.n_rows, memory.n_cols);

    BackwardWithMemory(std::move(output),
                       std::move(memory),
                       std::move(gy),
                       std::move(g),
                       std::move(gM));
  }

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
  template<typename eT>
  void Backward(const arma::Mat<eT>&& output,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g)
  {
    arma::mat dM;
    arma::mat memory = arma::ones<arma::mat>(numMem, memSize);
    BackwardWithMemory(std::move(output), std::move(memory),
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

  //! Number of memory locations.
  size_t numMem;

  //! Memory size in each memory block.
  size_t memSize;

  //! Shifting size used in circular convolution.
  size_t shiftSize;

  //! Linear layer to generate erase and add vectors.
  LayerTypes inputToLinear;

  //! Non linearity for Add operation.
  LayerTypes addGate;

  //! Non linearity for Erase operation.
  LayerTypes eraseGate;

  //! Memory head to generate write weights.
  LayerTypes writeHead;

  //! Locally stored error for linear layer.
  arma::mat prevError;

  //! Locally stored error for write head.
  arma::mat dWriteHead;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored list of network modules.
  std::vector<LayerTypes> network;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class WriteMemory

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "write_memory_impl.hpp"

#endif
