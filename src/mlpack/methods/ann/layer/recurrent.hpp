/**
 * @file methods/ann/layer/recurrent.hpp
 * @author Marcus Edel
 *
 * Definition of the Recurrent class.
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
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat,
>
class Recurrent : public Layer<InputType, OutputType>
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
  void Forward(const InputType& input, OutputType& output);

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

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& /* gradient */);

  //! Get the model modules.
  std::vector<Layer<InputType, OutputType>*>& Model() { return network; }

    //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the parameters.
  OutputType const& Parameters() const { return parameters; }
  //! Modify the parameters.
  OutputType& Parameters() { return parameters; }

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the gradient.
  OutputType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputType& Gradient() { return gradient; }

  //! Get the number of steps to backpropagate through time.
  size_t const& Rho() const { return rho; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored delete visitor module object.
  DeleteVisitor deleteVisitor;

  //! Locally-stored start module.
  Layer<InputType, OutputType>* startModule;

  //! Locally-stored input module.
  Layer<InputType, OutputType>* inputModule;

  //! Locally-stored feedback module.
  Layer<InputType, OutputType>* feedbackModule;

  //! Locally-stored transfer module.
  Layer<InputType, OutputType>* transferModule;

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
  OutputType parameters;

  //! Locally-stored initial module.
  Sequential<InputType, OutputType>* initialModule;

  //! Locally-stored recurrent module.
  Sequential<InputType, OutputType>* recurrentModule;

  //! Locally-stored model modules.
  std::vector<Layer<InputType, OutputType>*> network;

  //! Locally-stored merge module.
  AddMerge<InputType, OutputType>* mergeModule;

  //! Locally-stored feedback output parameters.
  std::vector<OutputType> feedbackOutputParameter;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored gradient object.
  OutputType gradient;

  //! Locally-stored output parameter object.
  OutputType outputParameter;

  //! Locally-stored recurrent error parameter.
  OutputType recurrentError;
}; // class Recurrent

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "recurrent_impl.hpp"

#endif
