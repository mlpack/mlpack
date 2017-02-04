/**
 * @file recurrent_attention_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the RecurrentAttention class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_IMPL_HPP

// In case it hasn't yet been included.
#include "recurrent_attention.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
template<typename RNNModuleType, typename ActionModuleType>
RecurrentAttention<InputDataType, OutputDataType>::RecurrentAttention(
    const size_t outSize,
    const RNNModuleType& rnn,
    const ActionModuleType& action,
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

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void RecurrentAttention<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
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

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void RecurrentAttention<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */,
    arma::Mat<eT>&& gy,
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

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void RecurrentAttention<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& /* input */,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  size_t offset = 0;
  offset += boost::apply_visitor(GradientUpdateVisitor(
      std::move(attentionGradient), offset), rnnModule);
  boost::apply_visitor(GradientUpdateVisitor(
      std::move(attentionGradient), offset), actionModule);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void RecurrentAttention<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(rho, "rho");
  ar & data::CreateNVP(outSize, "outSize");
  ar & data::CreateNVP(forwardStep, "forwardStep");
  ar & data::CreateNVP(backwardStep, "backwardStep");
}

} // namespace ann
} // namespace mlpack

#endif
