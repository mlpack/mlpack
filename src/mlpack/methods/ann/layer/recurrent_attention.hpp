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
<<<<<<< HEAD
<<<<<<< HEAD
                     const size_t rho);
=======
                     const size_t rho) :
      outSize(outSize),
      rnnModule(new RNNModuleType(rnn)),
      actionModule(new ActionModuleType(action)),
      rho(rho),
      forwardStep(0),
      backwardStep(0),
      deterministic(false)
  {
    network.push_back(rnnModule);
    network.push_back(actionModule);
  }
>>>>>>> Refactor neural visual attention modules.
=======
                     const size_t rho);
>>>>>>> Split layer modules into definition and implementation.

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
<<<<<<< HEAD
<<<<<<< HEAD
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output);
=======
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    // Initialize the action input.
    if (initialInput.is_empty())
    {
      initialInput = arma::zeros(outSize, input.n_cols);
    }

    // Propagate through the action and recurrent module.
    for (forwardStep = 0; forwardStep < rho; ++forwardStep)
    {
      if (forwardStep == 0)
      {
        boost::apply_visitor(ForwardVisitor(std::move(initialInput), std::move(
            boost::apply_visitor(outputParameterVisitor, actionModule))),
            actionModule);
      }
      else
      {
        boost::apply_visitor(ForwardVisitor(std::move(boost::apply_visitor(
            outputParameterVisitor, rnnModule)), std::move(boost::apply_visitor(
            outputParameterVisitor, actionModule))), actionModule);
      }

      // Initialize the glimpse input.
      arma::mat glimpseInput = arma::zeros(input.n_elem, 2);
      glimpseInput.col(0) = input;
      glimpseInput.submat(0, 1, boost::apply_visitor(outputParameterVisitor,
          actionModule).n_elem - 1, 1) = boost::apply_visitor(
          outputParameterVisitor, actionModule);

      boost::apply_visitor(ForwardVisitor(std::move(glimpseInput),
          std::move(boost::apply_visitor(outputParameterVisitor, rnnModule))),
          rnnModule);

      // Save the output parameter when training the module.
      if (!deterministic)
      {
        for (size_t l = 0; l < network.size(); ++l)
        {
          boost::apply_visitor(SaveOutputParameterVisitor(
              std::move(moduleOutputParameter)), network[l]);
        }
      }
    }

    output = boost::apply_visitor(outputParameterVisitor, rnnModule);

    forwardStep = 0;
    backwardStep = 0;
  }
>>>>>>> Refactor neural visual attention modules.
=======
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output);
>>>>>>> Split layer modules into definition and implementation.

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
<<<<<<< HEAD
<<<<<<< HEAD
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    if (intermediateGradient.is_empty() && backwardStep == 0)
    {
      // Initialize the attention gradients.
      size_t weights = boost::apply_visitor(weightSizeVisitor, rnnModule) +
          boost::apply_visitor(weightSizeVisitor, actionModule);

      intermediateGradient = arma::zeros(weights, 1);
      attentionGradient = arma::zeros(weights, 1);

      // Initialize the action error.
      actionError = arma::zeros(
        boost::apply_visitor(outputParameterVisitor, actionModule).n_rows,
        boost::apply_visitor(outputParameterVisitor, actionModule).n_cols);
    }

    // Propagate the attention gradients.
    if (backwardStep == 0)
    {
      size_t offset = 0;
      offset += boost::apply_visitor(GradientSetVisitor(
          std::move(intermediateGradient), offset), rnnModule);
      boost::apply_visitor(GradientSetVisitor(
          std::move(intermediateGradient), offset), actionModule);

      attentionGradient.zeros();
    }

    // Back-propagate through time.
    for (; backwardStep < rho; backwardStep++)
    {
      if (backwardStep == 0)
      {
        recurrentError = gy;
      }
      else
      {
        recurrentError = actionDelta;
      }

      for (size_t l = 0; l < network.size(); ++l)
      {
        boost::apply_visitor(LoadOutputParameterVisitor(
           std::move(moduleOutputParameter)), network[network.size() - 1 - l]);
      }

      if (backwardStep == (rho - 1))
      {
        boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
            outputParameterVisitor, actionModule)), std::move(actionError),
            std::move(actionDelta)), actionModule);
      }
      else
      {
        boost::apply_visitor(BackwardVisitor(std::move(initialInput),
            std::move(actionError), std::move(actionDelta)), actionModule);
      }

      boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, rnnModule)), std::move(recurrentError),
          std::move(rnnDelta)), rnnModule);

      if (backwardStep == 0)
      {
        g = rnnDelta.col(1);
      }
      else
      {
        g += rnnDelta.col(1);
      }

      IntermediateGradient();
    }
  }
>>>>>>> Refactor neural visual attention modules.
=======
                arma::Mat<eT>&& g);
>>>>>>> Split layer modules into definition and implementation.

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
<<<<<<< HEAD
<<<<<<< HEAD
                arma::Mat<eT>&& /* gradient */);
=======
                arma::Mat<eT>&& /* gradient */)
  {
    size_t offset = 0;
    offset += boost::apply_visitor(GradientUpdateVisitor(
        std::move(attentionGradient), offset), rnnModule);
    boost::apply_visitor(GradientUpdateVisitor(
        std::move(attentionGradient), offset), actionModule);
  }
>>>>>>> Refactor neural visual attention modules.
=======
                arma::Mat<eT>&& /* gradient */);
>>>>>>> Split layer modules into definition and implementation.

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
<<<<<<< HEAD
<<<<<<< HEAD
  void Serialize(Archive& ar, const unsigned int /* version */);
=======
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(rho, "rho");
    ar & data::CreateNVP(outSize, "outSize");
    ar & data::CreateNVP(forwardStep, "forwardStep");
    ar & data::CreateNVP(backwardStep, "backwardStep");
  }
>>>>>>> Refactor neural visual attention modules.
=======
  void Serialize(Archive& ar, const unsigned int /* version */);
>>>>>>> Split layer modules into definition and implementation.

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

<<<<<<< HEAD
<<<<<<< HEAD
// Include implementation.
#include "recurrent_attention_impl.hpp"

=======
>>>>>>> Refactor neural visual attention modules.
=======
// Include implementation.
#include "recurrent_attention_impl.hpp"

>>>>>>> Split layer modules into definition and implementation.
#endif
