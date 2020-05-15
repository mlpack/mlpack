/**
 * @file methods/ann/layer/recurrent.hpp
 * @author Marcus Edel
 *
 * Definition of the LinearLayer class also known as fully-connected layer or
 * affine transformation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_HPP

#include <mlpack/core.hpp>

#include "../visitor/delete_visitor.hpp"
#include "../visitor/delta_visitor.hpp"
#include "../visitor/copy_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"

#include "layer_types.hpp"
#include "add_merge.hpp"
#include "sequential.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the RecurrentLayer class. Recurrent layers can be used
 * similarly to feed-forward layers.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename... CustomLayers
>
class Recurrent
{
 public:
  /**
   * Default constructor---this will create a Recurrent object that can't be
   * used, so be careful!  Make sure to set all the parameters before use.
   */
  Recurrent();

  //! Copy constructor.
  Recurrent(const Recurrent&);

  /**
   * Create the Recurrent object using the specified modules.
   *
   * @param start The start module.
   * @param input The input module.
   * @param feedback The feedback module.
   * @param transfer The transfer module.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  template<typename StartModuleType,
           typename InputModuleType,
           typename FeedbackModuleType,
           typename TransferModuleType>
  Recurrent(const StartModuleType& start,
            const InputModuleType& input,
            const FeedbackModuleType& feedback,
            const TransferModuleType& transfer,
            const size_t rho);

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
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
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
                arma::Mat<eT>& /* gradient */);

  //! Get the model modules.
  std::vector<LayerTypes<CustomLayers...> >& Model() { return network; }

    //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the parameters.
  OutputDataType const& Parameters() const { return parameters; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return parameters; }

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

  //! Get the number of steps to backpropagate through time.
  size_t const& Rho() const { return rho; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored delete visitor module object.
  DeleteVisitor deleteVisitor;

  //! Locally-stored copy visitor
  CopyVisitor<CustomLayers...> copyVisitor;

  //! Locally-stored start module.
  LayerTypes<CustomLayers...> startModule;

  //! Locally-stored input module.
  LayerTypes<CustomLayers...> inputModule;

  //! Locally-stored feedback module.
  LayerTypes<CustomLayers...> feedbackModule;

  //! Locally-stored transfer module.
  LayerTypes<CustomLayers...> transferModule;

  //! Number of steps to backpropagate through time (BPTT).
  size_t rho;

  //! Locally-stored number of forward steps.
  size_t forwardStep;

  //! Locally-stored number of backward steps.
  size_t backwardStep;

  //! Locally-stored number of gradient steps.
  size_t gradientStep;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! To know whether this object allocated memory. We need this to know
  //! whether we should delete the metric member variable in the destructor.
  bool ownsLayer;

  //! Locally-stored weight object.
  OutputDataType parameters;

  //! Locally-stored initial module.
  LayerTypes<CustomLayers...> initialModule;

  //! Locally-stored recurrent module.
  LayerTypes<CustomLayers...> recurrentModule;

  //! Locally-stored model modules.
  std::vector<LayerTypes<CustomLayers...> > network;

  //! Locally-stored merge module.
  LayerTypes<CustomLayers...> mergeModule;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored feedback output parameters.
  std::vector<arma::mat> feedbackOutputParameter;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored recurrent error parameter.
  arma::mat recurrentError;
}; // class Recurrent

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "recurrent_impl.hpp"

#endif
