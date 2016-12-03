/**
 * @file recurrent.hpp
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
#include <boost/ptr_container/ptr_vector.hpp>

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
    typename OutputDataType = arma::mat
>
class Recurrent
{
 public:
  /**
   * Create the Recurrent object using the specified modules.
   *
   * @param start The start module.
   * @param start The input module.
   * @param start The feedback module.
   * @param start The transfer module.
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
<<<<<<< HEAD
            const size_t rho);
=======
            const size_t rho) :
      startModule(new StartModuleType(start)),
      inputModule(new InputModuleType(input)),
      feedbackModule(new FeedbackModuleType(feedback)),
      transferModule(new TransferModuleType(transfer)),
      rho(rho),
      forwardStep(0),
      backwardStep(0),
      gradientStep(0),
      deterministic(false)

  {
    initialModule = new Sequential<>();
    mergeModule = new AddMerge<>();
    recurrentModule = new Sequential<>(false);

    boost::apply_visitor(AddVisitor(inputModule), initialModule);
    boost::apply_visitor(AddVisitor(startModule), initialModule);
    boost::apply_visitor(AddVisitor(transferModule), initialModule);

    boost::apply_visitor(weightSizeVisitor, startModule);
    boost::apply_visitor(weightSizeVisitor, inputModule);
    boost::apply_visitor(weightSizeVisitor, feedbackModule);
    boost::apply_visitor(weightSizeVisitor, transferModule);

    boost::apply_visitor(AddVisitor(inputModule), mergeModule);
    boost::apply_visitor(AddVisitor(feedbackModule), mergeModule);
    boost::apply_visitor(AddVisitor(mergeModule), recurrentModule);
    boost::apply_visitor(AddVisitor(transferModule), recurrentModule);

    network.push_back(initialModule);
    network.push_back(mergeModule);
    network.push_back(feedbackModule);
    network.push_back(recurrentModule);
  }
>>>>>>> Refactor ann layer.

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
<<<<<<< HEAD
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output);
=======
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    if (forwardStep == 0)
    {
      boost::apply_visitor(ForwardVisitor(std::move(input), std::move(output)),
          initialModule);
    }
    else
    {
      boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
          boost::apply_visitor(outputParameterVisitor, inputModule))),
          inputModule);

      boost::apply_visitor(ForwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, transferModule)), std::move(
          boost::apply_visitor(outputParameterVisitor, feedbackModule))),
          feedbackModule);

      boost::apply_visitor(ForwardVisitor(std::move(input), std::move(output)),
          recurrentModule);
    }

    output = boost::apply_visitor(outputParameterVisitor, transferModule);

    // Save the feedback output parameter when training the module.
    if (!deterministic)
    {
      feedbackOutputParameter.push_back(output);
    }

    forwardStep++;
    if (forwardStep == rho)
    {
      forwardStep = 0;
      backwardStep = 0;

      if (!recurrentError.is_empty())
      {
        recurrentError.zeros();
      }
    }
  }
>>>>>>> Refactor ann layer.

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
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    if (!recurrentError.is_empty())
    {
      recurrentError += gy;
    }
    else
    {
      recurrentError = gy;
    }

    if (backwardStep < (rho - 1))
    {
      boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, recurrentModule)), std::move(recurrentError),
          std::move(boost::apply_visitor(deltaVisitor, recurrentModule))),
          recurrentModule);

      boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, inputModule)), std::move(
          boost::apply_visitor(deltaVisitor, recurrentModule)), std::move(g)),
          inputModule);

      boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, feedbackModule)), std::move(
          boost::apply_visitor(deltaVisitor, recurrentModule)), std::move(
          boost::apply_visitor(deltaVisitor, feedbackModule))),feedbackModule);
    }
    else
    {
      boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
          outputParameterVisitor, initialModule)), std::move(recurrentError),
          std::move(g)), initialModule);
    }

    recurrentError = boost::apply_visitor(deltaVisitor, feedbackModule);
    backwardStep++;
  }
>>>>>>> Refactor ann layer.

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Mat<eT>&& input,
                arma::Mat<eT>&& error,
<<<<<<< HEAD
                arma::Mat<eT>&& /* gradient */);
=======
                arma::Mat<eT>&& /* gradient */)
  {
    if (gradientStep < (rho - 1))
    {
      boost::apply_visitor(GradientVisitor(std::move(input), std::move(error)),
          recurrentModule);

      boost::apply_visitor(GradientVisitor(std::move(input), std::move(
          boost::apply_visitor(deltaVisitor, mergeModule))), inputModule);

      boost::apply_visitor(GradientVisitor(std::move(
          feedbackOutputParameter[feedbackOutputParameter.size() - 2 -
          gradientStep]), std::move(boost::apply_visitor(deltaVisitor,
          mergeModule))), feedbackModule);
    }
    else
    {
      boost::apply_visitor(GradientZeroVisitor(), recurrentModule);
      boost::apply_visitor(GradientZeroVisitor(), inputModule);
      boost::apply_visitor(GradientZeroVisitor(), feedbackModule);

      boost::apply_visitor(GradientVisitor(std::move(input), std::move(
          boost::apply_visitor(deltaVisitor, startModule))), initialModule);
    }

    gradientStep++;
    if (gradientStep == rho)
    {
      gradientStep = 0;
      feedbackOutputParameter.clear();
    }
  }
>>>>>>> Refactor ann layer.

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
  void Serialize(Archive& ar, const unsigned int /* version */);
=======
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(rho, "rho");
  }
>>>>>>> Refactor ann layer.

 private:
  //! Locally-stored start module.
  LayerTypes startModule;

  //! Locally-stored input module.
  LayerTypes inputModule;

  //! Locally-stored feedback module.
  LayerTypes feedbackModule;

  //! Locally-stored transfer module.
  LayerTypes transferModule;

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
}; // class Recurrent

} // namespace ann
} // namespace mlpack

<<<<<<< HEAD
// Include implementation.
#include "recurrent_impl.hpp"

=======
>>>>>>> Refactor ann layer.
#endif
