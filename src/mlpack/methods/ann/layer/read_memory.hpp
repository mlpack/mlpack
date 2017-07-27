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
#ifndef MLPACK_METHODS_ANN_LAYER_READ_MEMORY_HPP
#define MLPACK_METHODS_ANN_LAYER_READ_MEMORY_HPP

#include <mlpack/prereqs.hpp>

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
class ReadMemory
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
  ReadMemory();

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
    arma::mat memory = arma::ones<arma::mat>(3, 5);
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
                          const arma::Mat<eT>&& input,
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
    arma::mat memory = arma::ones<arma::mat>(3, 5);
    BackwardWithMemory(std::move(output), std::move(input), std::move(memory),
        std::move(gy), std::move(g), std::move(dM));
  }

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

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class LSTM

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "read_memory_impl.hpp"

#endif
