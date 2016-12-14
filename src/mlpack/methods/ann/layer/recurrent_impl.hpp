/**
 * @file recurrent_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the LinearLayer class also known as fully-connected layer
 * or affine transformation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_IMPL_HPP

// In case it hasn't yet been included.
#include "recurrent.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
template<
    typename StartModuleType,
    typename InputModuleType,
    typename FeedbackModuleType,
    typename TransferModuleType
>
Recurrent<InputDataType, OutputDataType>::Recurrent(
    const StartModuleType& start,
    const InputModuleType& input,
    const FeedbackModuleType& feedback,
    const TransferModuleType& transfer,
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

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Recurrent<InputDataType, OutputDataType>::Forward(
    arma::Mat<eT>&& input, arma::Mat<eT>&& output)
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

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Recurrent<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
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

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Recurrent<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
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

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Recurrent<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(rho, "rho");
}

} // namespace ann
} // namespace mlpack

#endif
