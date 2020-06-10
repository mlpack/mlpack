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

#include "../visitor/delete_visitor.hpp"
#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"

#include <boost/ptr_container/ptr_vector.hpp>

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Concat class. The Concat class works as a
 * feed-forward fully connected network container which plugs various layers
 * together.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam CustomLayers Additional custom layers if required.
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename... CustomLayers
>
class Concat
{
 public:
  /**
   * Create the Concat object using the specified parameters.
   *
   * @param model Expose all network modules.
   * @param run Call the Forward/Backward method before the output is merged.
   */
  Concat(const bool model = false,
         const bool run = true);

  /**
   * Create the Concat object using the specified parameters.
   *
   * @param inputSize A vector denoting input size of each layer added.
   * @param axis Concat axis.
   * @param model Expose all network modules.
   * @param run Call the Forward/Backward method before the output is merged.
   */
  Concat(arma::Row<size_t>& inputSize,
         const size_t axis,
         const bool model = false,
         const bool run = true);

  /**
   * Destroy the layers held by the model.
   */
  ~Concat();

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
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /**
   * This is the overload of Backward() that runs only a specific layer with
   * the given input.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   * @param index The index of the layer to run.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g,
                const size_t index);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& /* gradient */);

  /*
   * This is the overload of Gradient() that runs a specific layer with the
   * given input.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   * @param The index of the layer to run.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& gradient,
                const size_t index);

  /*
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args) { network.push_back(new LayerType(args...)); }

  /*
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(LayerTypes<CustomLayers...> layer) { network.push_back(layer); }

  //! Return the model modules.
  std::vector<LayerTypes<CustomLayers...> >& Model()
  {
    if (model)
    {
      return network;
    }

    return empty;
  }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameters; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameters; }

  //! Get the value of run parameter.
  bool Run() const { return run; }
  //! Modify the value of run parameter.
  bool& Run() { return run; }

  arma::mat const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  arma::mat& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  arma::mat const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  arma::mat& OutputParameter() { return outputParameter; }

  //! Get the delta.e
  arma::mat const& Delta() const { return delta; }
  //! Modify the delta.
  arma::mat& Delta() { return delta; }

  //! Get the gradient.
  arma::mat const& Gradient() const { return gradient; }
  //! Modify the gradient.
  arma::mat& Gradient() { return gradient; }

  //! Get the axis of concatenation.
  size_t const& ConcatAxis() const { return axis; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */);

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
  std::vector<LayerTypes<CustomLayers...> > network;

  //! Locally-stored model parameters.
  arma::mat parameters;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! Locally-stored empty list of modules.
  std::vector<LayerTypes<CustomLayers...> > empty;

  //! Locally-stored delta object.
  arma::mat delta;

  //! Locally-stored input parameter object.
  arma::mat inputParameter;

  //! Locally-stored output parameter object.
  arma::mat outputParameter;

  //! Locally-stored gradient object.
  arma::mat gradient;
}; // class Concat

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concat_impl.hpp"

#endif
