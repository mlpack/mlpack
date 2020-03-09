/**
 * @file radial_basis_function.hpp
 * @author Himanshu Pathak
 *
 * Definition of the Radial Basis Function class.
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{,
 *    title   = {},
 *    author  = {},
 *    journal = {},
 *    year    = {}
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

#include "layer_types.hpp"
#include "add_merge.hpp"
#include "sequential.hpp"


namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * In Radial Basis Function, the expressiveness
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class RBF
{
 public:
  //! Create the RBF object.
  RBF();


  /**
   * Create the Radial Basis Function layer object using the specified
   * parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
  RBF(const size_t inSize,
    const size_t outSize);

  /**
   * Delete the NeuralTuringMachine and the layers it holds.
   */
  ~RBF();

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
  //! Locally-stored the learnable centre of the shape.
  InputDataType centres;

  //! Locally-stored the learnable scaling factor of the shape.
  InputDataType sigmas;

  //! Locally-stored the outeput distances of the shape.
  InputDataType distances;

  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

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
}; // class RBF

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "radial_basis_function_impl.hpp"

#endif
