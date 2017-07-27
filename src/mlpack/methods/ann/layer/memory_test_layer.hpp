/**
 * @file memory_test_layer.hpp
 * @author Sumedh Ghaisas
 *
 * Definition of MemoryTest class which acts as an abstraction to test layers
 * which handle memory.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEMORY_TEST_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_MEMORY_TEST_LAYER_HPP

#include <list>

#include <mlpack/prereqs.hpp>

#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a MemoryTest layer.
 *
 * This layer is used in testing layers which handle memory.
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
class MemoryTest
{
 public:
  /**
   * Create the MemoryTest layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  MemoryTest(const size_t inSize,
             const size_t outSize,
             const size_t numMme,
             const size_t memSize,
             LayerTypes testLayer);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output);

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
  void Backward(const arma::Mat<eT>&& /* output */,
                const arma::Mat<eT>&& input,
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

  //! Number of memory locations.
  size_t numMem;

  //! Size of each memory location.
  size_t memSize;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! The layer to be tested.
  LayerTypes testLayer;

  //! Memory init layer.
  LayerTypes initMem;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored list of network modules.
  std::vector<LayerTypes> network;

  //! Locally-stored previous error.
  arma::mat prevError;

  //! Error with respect to memory.
  arma::mat dMem;

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
}; // class GRU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "memory_test_layer_impl.hpp"

#endif
