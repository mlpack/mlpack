/**
 * @file methods/ann/layer/add_merge.hpp
 * @author Marcus Edel
 *
 * Definition of the AddMerge module which accumulates the output of the given
 * modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_MERGE_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_MERGE_HPP

#include <mlpack/prereqs.hpp>

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the AddMerge module class. The AddMerge class accumulates
 * the output of various modules.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam CustomLayers Additional custom layers that can be added.
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class AddMerge : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the AddMerge object using the specified parameters.
   *
   * @param model Expose all the network modules.
   * @param run Call the Forward/Backward method before the output is merged.
   */
  AddMerge(const bool model = false, const bool run = true);

  /**
   * Create the AddMerge object using the specified parameters.
   *
   * @param model Expose all the network modules.
   * @param run Call the Forward/Backward method before the output is merged.
   * @param ownsLayers Delete the layers when this is deallocated.
   */
  AddMerge(const bool model, const bool run, const bool ownsLayers);

  //! Destructor to release allocated memory.
  ~AddMerge();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param * (input) Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& /* input */, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
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

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  /*
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
  void Add(Layer<InputType, OutputType>* layer) { network.push_back(layer); }

  //! Get the input parameter.
  InputType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Return the model modules.
  std::vector<Layer<InputType, OutputType>*>& Model()
  {
    if (model)
    {
      return network;
    }

    return empty;
  }

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the value of run parameter.
  bool Run() const { return run; }
  //! Modify the value of run parameter.
  bool& Run() { return run; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Parameter which indicates if the modules should be exposed.
  bool model;

  //! Parameter which indicates if the Forward/Backward method should be called
  //! before merging the output.
  bool run;

  //! We need this to know whether we should delete the internally-held layers
  //! in the destructor.
  bool ownsLayers;

  //! Locally-stored network modules.
  std::vector<Layer<InputType, OutputType>*> network;

  //! Locally-stored empty list of modules.
  std::vector<Layer<InputType, OutputType>*> empty;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored gradient object.
  OutputType gradient;

  //! Locally-stored input parameter object.
  InputType inputParameter;

  //! Locally-stored output parameter object.
  OutputType outputParameter;

  //! Locally-stored weight object.
  OutputType weights;
}; // class AddMerge

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "add_merge_impl.hpp"

#endif
