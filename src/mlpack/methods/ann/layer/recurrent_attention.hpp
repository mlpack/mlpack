/**
 * @file recurrent_attention.hpp
 * @author Marcus Edel
 *
 * Definition of the RecurrentAttention class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_HPP

#include <mlpack/core.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "layer_types.hpp"
#include "add_merge.hpp"
#include "sequential.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class implements the Recurrent Model for Visual Attention, using a
 * variety of possible layer implementations.
 *
 * For more information, see the following paper.
 *
 * @code
 * @article{MnihHGK14,
 *   title={Recurrent Models of Visual Attention},
 *   author={Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu},
 *   journal={CoRR},
 *   volume={abs/1406.6247},
 *   year={2014}
 * }
 * @endcode
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
class RecurrentAttention
{
 public:
  /**
   * Create the RecurrentAttention object using the specified modules.
   *
   * @param start The module output size.
   * @param start The recurrent neural network module.
   * @param start The action module.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  template<typename RNNModuleType, typename ActionModuleType>
  RecurrentAttention(const size_t outSize,
                     const RNNModuleType& rnn,
                     const ActionModuleType& action,
                     const size_t rho);

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
  void Backward(const arma::Mat<eT>&& /* input */,
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
  void Gradient(arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& /* error */,
                arma::Mat<eT>&& /* gradient */);

  //! Get the model modules.
  std::vector<LayerTypes>& Model() { return network; }

    //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the parameters.
  OutputDataType const& Parameters() const { return parameters; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return parameters; }

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

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Calculate the gradient of the attention module.
  void IntermediateGradient()
  {
    intermediateGradient.zeros();

    // Gradient of the action module.
    if (backwardStep == (rho - 1))
    {
      boost::apply_visitor(GradientVisitor(std::move(initialInput),
          std::move(actionError)), actionModule);
    }
    else
    {
      boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, actionModule)), std::move(actionError)),
          actionModule);
    }

    // Gradient of the recurrent module.
    boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, rnnModule)), std::move(recurrentError)),
        rnnModule);

    attentionGradient += intermediateGradient;
  }

  //! Locally-stored module output size.
  size_t outSize;

  //! Locally-stored start module.
  LayerTypes rnnModule;

  //! Locally-stored input module.
  LayerTypes actionModule;

  //! Number of steps to backpropagate through time (BPTT).
  size_t rho;

  //! Locally-stored number of forward steps.
  size_t forwardStep;

  //! Locally-stored number of backward steps.
  size_t backwardStep;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Locally-stored weight object.
  OutputDataType parameters;

  //! Locally-stored initial module.
  LayerTypes initialModule;

  //! Locally-stored recurrent module.
  LayerTypes recurrentModule;

  //! Locally-stored model modules.
  std::vector<LayerTypes> network;

  //! Locally-stored merge module.
  LayerTypes mergeModule;

  //! Locally-stored weight size visitor.
  WeightSizeVisitor weightSizeVisitor;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored feedback output parameters.
  std::vector<arma::mat> feedbackOutputParameter;

  //! List of all module parameters for the backward pass (BBTT).
  std::vector<arma::mat> moduleOutputParameter;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored recurrent error parameter.
  arma::mat recurrentError;

  //! Locally-stored action error parameter.
  arma::mat actionError;

  //! Locally-stored action delta.
  arma::mat actionDelta;

  //! Locally-stored recurrent delta.
  arma::mat rnnDelta;

  //! Locally-stored initial action input.
  arma::mat initialInput;

  //! Locally-stored reset visitor.
  ResetVisitor resetVisitor;

  //! Locally-stored attention gradient.
  arma::mat attentionGradient;

  //! Locally-stored intermediate gradient for the attention module.
  arma::mat intermediateGradient;
}; // class RecurrentAttention

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "recurrent_attention_impl.hpp"

#endif
