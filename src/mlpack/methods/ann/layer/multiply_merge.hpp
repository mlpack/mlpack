/**
 * @file methods/ann/layer/multiply_merge.hpp
 * @author Haritha Nair
 *
 * Definition of the MultiplyMerge module which multiplies the output of the
 * given modules element-wise.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_HPP

#include <mlpack/prereqs.hpp>

#include "../visitor/delete_visitor.hpp"
#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the MultiplyMerge module class. The MultiplyMerge class
 * multiplies the output of various modules element-wise.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam CustomLayers Additional custom layers that can be added.
 */
template<
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename... CustomLayers
>
class MultiplyMerge
{
 public:
  /**
   * Create the MultiplyMerge object using the specified parameters.
   *
   * @param model Expose all the network modules.
   * @param run Call the Forward/Backward method before the output is merged.
   */
  MultiplyMerge(const bool model = false, const bool run = true);

  //! Destructor to release allocated memory.
  ~MultiplyMerge();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param * (input) Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType& /* input */, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f, using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& gradient);

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

  //! Return the model modules.
  std::vector<LayerTypes<CustomLayers...> >& Model()
  {
    if (model)
    {
      return network;
    }

    return empty;
  }

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar);

 private:
  //! Parameter which indicates if the modules should be exposed.
  bool model;

  //! Parameter which indicates if the Forward/Backward method should be called
  //! before merging the output.
  bool run;

  //! We need this to know whether we should delete the layer in the destructor.
  bool ownsLayer;

  //! Locally-stored network modules.
  std::vector<LayerTypes<CustomLayers...> > network;

  //! Locally-stored empty list of modules.
  std::vector<LayerTypes<CustomLayers...> > empty;

  //! Locally-stored delete visitor module object.
  DeleteVisitor deleteVisitor;

  //! Locally-stored output parameter visitor module object.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delta visitor module object.
  DeltaVisitor deltaVisitor;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored weight object.
  OutputDataType weights;
}; // class MultiplyMerge

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "multiply_merge_impl.hpp"

#endif
