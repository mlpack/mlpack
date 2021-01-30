/**
 * @file methods/ann/layer/concat.hpp
 * @author Marcus Edel
 * @author Mehul Kumar Nirala
 *
 * Definition of the Concat class, which acts as a concatenation container.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Concat class. The Concat class works as a
 * feed-forward fully connected network container which plugs various layers
 * together.
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
class ConcatType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the Concat object using the specified parameters.
   *
   * @param model Expose all network modules.
   * @param run Call the Forward/Backward method before the output is merged.
   */
  ConcatType(const bool model = false,
             const bool run = true);

  /**
   * Create the Concat object using the specified parameters.
   *
   * @param inputSize A vector denoting input size of each layer added.
   * @param axis Concat axis.
   * @param model Expose all network modules.
   * @param run Call the Forward/Backward method before the output is merged.
   */
  ConcatType(arma::Row<size_t>& inputSize,
             const size_t axis,
             const bool model = false,
             const bool run = true);

  /**
   * Destroy the layers held by the model.
   */
  ~ConcatType();

	//! Clone the ConcatType object. This handles polymorphism correctly.
	ConcatType* Clone() const { return new ConcatType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /**
   * This is the overload of Backward() that runs only a specific layer with
   * the given input.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   * @param index The index of the layer to run.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g,
                const size_t index);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& /* input */,
                const OutputType& error,
                OutputType& /* gradient */);

  /**
   * This is the overload of Gradient() that runs a specific layer with the
   * given input.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   * @param The index of the layer to run.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient,
                const size_t index);

  /**
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args) { network.push_back(new LayerType(args...)); }

  /**
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(Layer<>* layer) { network.push_back(layer); }

  //! Return the model modules.
  std::vector<Layer<>*>& Model()
  {
    if (model)
    {
      return network;
    }

    return empty;
  }

  //! Return the initial point for the optimization.
  const OutputType& Parameters() const { return weights; }
  //! Modify the initial point for the optimization.
  OutputType& Parameters() { return weights; }

  //! Get the value of run parameter.
  bool Run() const { return run; }
  //! Modify the value of run parameter.
  bool& Run() { return run; }

  const InputType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  const OutputType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.e
  const OutputType& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the gradient.
  const OutputType& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputType& Gradient() { return gradient; }

  //! Get the axis of concatenation.
  const size_t& ConcatAxis() const { return axis; }

  //! Get the size of the weight matrix.
  size_t WeightSize() const { return 0; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar,  const uint32_t /* version */);

 private:
  //! Parameter which indicates the input size of modules.
  arma::Row<size_t> inputSize;

  //! Parameter which indicates the axis of concatenation.
  size_t axis;

  //! Parameter which indicates whether to use the axis of concatenation.
  bool useAxis;

  //! Parameter which indicates if the modules should be exposed.
  bool model;

  //! Parameter which indicates if the Forward/Backward method should be called
  //! before merging the output.
  bool run;

  //! Parameter to store channels.
  size_t channels;

  //! Locally-stored network modules.
  std::vector<Layer<>*> network;

  //! Locally-stored model weights.
  OutputType weights;

  //! Locally-stored empty list of modules.
  std::vector<Layer<>*> empty;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored input parameter object.
  InputType inputParameter;

  //! Locally-stored output parameter object.
  OutputType outputParameter;

  //! Locally-stored gradient object.
  OutputType gradient;
}; // class ConcatType.

// Standard Concat layer.
typedef ConcatType<arma::mat, arma::mat> Concat;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concat_impl.hpp"

#endif
