/**
 * @file rmva_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Recurrent Model for Visual Attention.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_RMVA_RMVA_IMPL_HPP
#define __MLPACK_METHODS_RMVA_RMVA_IMPL_HPP

// In case it hasn't been included yet.
#include "rmva.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<
  typename LocatorType,
  typename LocationSensorType,
  typename GlimpseSensorType,
  typename GlimpseType,
  typename StartType,
  typename FeedbackType,
  typename TransferType,
  typename ClassifierType,
  typename RewardPredictorType,
  typename InitializationRuleType,
  typename MatType
>
template<
    typename TypeLocator,
    typename TypeLocationSensor,
    typename TypeGlimpseSensor,
    typename TypeGlimpse,
    typename TypeStart,
    typename TypeFeedback,
    typename TypeTransfer,
    typename TypeClassifier,
    typename TypeRewardPredictor
>
RecurrentNeuralAttention<
  LocatorType,
  LocationSensorType,
  GlimpseSensorType,
  GlimpseType,
  StartType,
  FeedbackType,
  TransferType,
  ClassifierType,
  RewardPredictorType,
  InitializationRuleType,
  MatType
>::RecurrentNeuralAttention(TypeLocator&& locator,
                            TypeLocationSensor&& locationSensor,
                            TypeGlimpseSensor&& glimpseSensor,
                            TypeGlimpse&& glimpse,
                            TypeStart&& start,
                            TypeFeedback&& feedback,
                            TypeTransfer&& transfer,
                            TypeClassifier&& classifier,
                            TypeRewardPredictor&& rewardPredictor,
                            const size_t nStep,
                            InitializationRuleType initializeRule) :
    locator(std::forward<TypeLocator>(locator)),
    locationSensor(std::forward<TypeLocationSensor>(locationSensor)),
    glimpseSensor(std::forward<TypeGlimpseSensor>(glimpseSensor)),
    glimpse(std::forward<TypeGlimpse>(glimpse)),
    start(std::forward<TypeStart>(start)),
    feedback(std::forward<TypeFeedback>(feedback)),
    transfer(std::forward<TypeTransfer>(transfer)),
    classifier(std::forward<TypeClassifier>(classifier)),
    rewardPredictor(std::forward<TypeRewardPredictor>(rewardPredictor)),
    nStep(nStep),
    inputSize(0)
{
  // Set the network size.
  locatorSize = NetworkSize(this->locator);
  locationSensorSize = NetworkSize(this->locationSensor);
  glimpseSensorSize = NetworkSize(this->glimpseSensor);
  glimpseSize = NetworkSize(this->glimpse);
  feedbackSize = NetworkSize(this->feedback);
  transferSize = NetworkSize(this->transfer);
  classifierSize = NetworkSize(this->classifier);
  rewardPredictorSize = NetworkSize(this->rewardPredictor);
  startSize = NetworkSize(this->start);

  initializeRule.Initialize(parameter, locatorSize + locationSensorSize + glimpseSensorSize +
      glimpseSize + feedbackSize + transferSize + classifierSize + rewardPredictorSize + startSize, 1);

  // Set the network weights.
  NetworkWeights(initializeRule, parameter, this->locator);
  NetworkWeights(initializeRule, parameter, this->locationSensor, locatorSize);
  NetworkWeights(initializeRule, parameter, this->glimpseSensor, locatorSize +
      locationSensorSize);
  NetworkWeights(initializeRule, parameter, this->glimpse, locatorSize +
      locationSensorSize + glimpseSensorSize);
  NetworkWeights(initializeRule, parameter, this->feedback, locatorSize +
      locationSensorSize + glimpseSensorSize + glimpseSize);
  NetworkWeights(initializeRule, parameter, this->transfer, locatorSize +
      locationSensorSize + glimpseSensorSize + glimpseSize + feedbackSize);
  NetworkWeights(initializeRule, parameter, this->classifier, locatorSize +
      locationSensorSize + glimpseSensorSize + glimpseSize + feedbackSize +
      transferSize);
  NetworkWeights(initializeRule, parameter, this->rewardPredictor, locatorSize +
      locationSensorSize + glimpseSensorSize + glimpseSize + feedbackSize +
      transferSize + classifierSize);
  NetworkWeights(initializeRule, parameter, this->start, locatorSize +
      locationSensorSize + glimpseSensorSize + glimpseSize + feedbackSize +
      transferSize + classifierSize + rewardPredictorSize);

  rewardInput = arma::field<arma::mat>(2, 1);
}

template<
  typename LocatorType,
  typename LocationSensorType,
  typename GlimpseSensorType,
  typename GlimpseType,
  typename StartType,
  typename FeedbackType,
  typename TransferType,
  typename ClassifierType,
  typename RewardPredictorType,
  typename InitializationRuleType,
  typename MatType
>
template<template<typename> class OptimizerType>
void RecurrentNeuralAttention<
  LocatorType,
  LocationSensorType,
  GlimpseSensorType,
  GlimpseType,
  StartType,
  FeedbackType,
  TransferType,
  ClassifierType,
  RewardPredictorType,
  InitializationRuleType,
  MatType
>::Train(const arma::mat& predictors,
         const arma::mat& responses,
         OptimizerType<NetworkType>& optimizer)
{
  numFunctions = predictors.n_cols;
  this->predictors = predictors;
  this->responses = responses;

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<
  typename LocatorType,
  typename LocationSensorType,
  typename GlimpseSensorType,
  typename GlimpseType,
  typename StartType,
  typename FeedbackType,
  typename TransferType,
  typename ClassifierType,
  typename RewardPredictorType,
  typename InitializationRuleType,
  typename MatType
>
void RecurrentNeuralAttention<
  LocatorType,
  LocationSensorType,
  GlimpseSensorType,
  GlimpseType,
  StartType,
  FeedbackType,
  TransferType,
  ClassifierType,
  RewardPredictorType,
  InitializationRuleType,
  MatType
>::Predict(arma::mat& predictors, arma::mat& responses)
{
  deterministic = true;

  arma::mat responsesTemp;
  SinglePredict(arma::cube(predictors.colptr(0), 28, 28, 1), responsesTemp);

  responses = arma::mat(responsesTemp.n_elem, predictors.n_cols);
  responses.col(0) = responsesTemp.col(0);

  for (size_t i = 1; i < predictors.n_cols; i++)
  {
    SinglePredict(arma::cube(predictors.colptr(i), 28, 28, 1), responsesTemp);
    responses.col(i) = responsesTemp.col(0);
  }
}

template<
  typename LocatorType,
  typename LocationSensorType,
  typename GlimpseSensorType,
  typename GlimpseType,
  typename StartType,
  typename FeedbackType,
  typename TransferType,
  typename ClassifierType,
  typename RewardPredictorType,
  typename InitializationRuleType,
  typename MatType
>
double RecurrentNeuralAttention<
  LocatorType,
  LocationSensorType,
  GlimpseSensorType,
  GlimpseType,
  StartType,
  FeedbackType,
  TransferType,
  ClassifierType,
  RewardPredictorType,
  InitializationRuleType,
  MatType
>::Evaluate(const arma::mat& /* unused */,
            const size_t i,
            const bool deterministic)
{
  this->deterministic = deterministic;

  input = arma::cube(predictors.colptr(i), 28, 28, 1);
  target = arma::mat(responses.colptr(i), responses.n_rows, 1, false, true);

  // Get the locator input size.
  if (!inputSize)
  {
    inputSize = NetworkInputSize(locator);
  }

  glimpseSensorMatCounter = 0;
  glimpseSensorCubeCounter = 0;
  glimpseActivationsCounter = 0;
  locatorActivationsCounter = 0;
  locationSensorActivationsCounter = 0;
  glimpseSensorMatActivationsCounter = 0;
  glimpseSensorCubeActivationsCounter = 0;
  locationCounter = 0;
  feedbackActivationsCounter = 0;
  transferActivationsCounter = 0;

  // Reset networks.
  ResetParameter(locator);
  ResetParameter(locationSensor);
  ResetParameter(glimpseSensor);
  ResetParameter(glimpse);
  ResetParameter(feedback);
  ResetParameter(transfer);
  ResetParameter(classifier);
  ResetParameter(rewardPredictor);
  ResetParameter(start);

  // Reset activation storage.
  glimpseActivations.clear();
  locatorActivations.clear();
  locationSensorActivations.clear();
  glimpseSensorMatActivations.clear();
  glimpseSensorCubeActivations.clear();
  feedbackActivations.clear();
  transferActivations.clear();
  locatorInput.clear();
  location.clear();
  feedbackActivationsInput.clear();

  // Sample an initial starting actions by forwarding zeros through the locator.
  locatorInput.push_back(new arma::cube(arma::zeros<arma::cube>(inputSize, 1,
      input.n_slices)));

  // Forward pass throught the recurrent network.
  for (step = 0; step < nStep; step++)
  {
    // Locator forward pass.
    Forward(locatorInput.back(), locator);
    SaveActivations(locatorActivations, locator, locatorActivationsCounter);

    // Location sensor forward pass.
    Forward(std::get<std::tuple_size<LocatorType>::value - 1>(
        locator).OutputParameter(), locationSensor);
    SaveActivations(locationSensorActivations, locationSensor,
        locationSensorActivationsCounter);

    // Set the location parameter for all layer that implement a Location
    // function e.g. GlimpseLayer.
    ResetLocation(std::get<std::tuple_size<LocatorType>::value - 1>(
        locator).OutputParameter(), glimpseSensor);

    // Save the location for the backward path.
    location.push_back(new arma::mat(std::get<std::tuple_size<
        LocatorType>::value - 1>(locator).OutputParameter()));

    // Glimpse sensor forward pass.
    Forward(input, glimpseSensor);
    SaveActivations(glimpseSensorMatActivations, glimpseSensorCubeActivations,
        glimpseSensorMatCounter, glimpseSensorCubeCounter, glimpseSensor);

    // Concat the parameter activation from the location sensor and
    // glimpse sensor.
    arma::mat concatLayerOutput = arma::join_cols(
        std::get<std::tuple_size<LocationSensorType>::value - 1>(
        locationSensor).OutputParameter(),
        std::get<std::tuple_size<GlimpseSensorType>::value - 1>(
        glimpseSensor).OutputParameter());

    // Glimpse forward pass.
    Forward(concatLayerOutput, glimpse);
    SaveActivations(glimpseActivations, glimpse, glimpseActivationsCounter);

    if (step == 0)
    {
      // Start forward pass.
      Forward(std::get<std::tuple_size<GlimpseType>::value - 1>(
          glimpse).OutputParameter(), start);

      // Transfer forward pass.
      Forward(std::get<std::tuple_size<StartType>::value - 1>(
          start).OutputParameter(), transfer);
      SaveActivations(transferActivations, transfer,
          transferActivationsCounter);
    }
    else
    {
      // Feedback forward pass.
      Forward(std::get<std::tuple_size<TransferType>::value - 1>(
          transfer).OutputParameter(), feedback);
      SaveActivations(feedbackActivations, feedback,
          feedbackActivationsCounter);

      feedbackActivationsInput.push_back(new arma::mat(
          std::get<std::tuple_size<TransferType>::value - 1>(
          transfer).OutputParameter().memptr(),
          std::get<std::tuple_size<TransferType>::value - 1>(
          transfer).OutputParameter().n_rows,
          std::get<std::tuple_size<TransferType>::value - 1>(
          transfer).OutputParameter().n_cols));

      arma::mat feedbackLayerOutput =
        std::get<std::tuple_size<GlimpseType>::value - 1>(
        glimpse).OutputParameter() +
        std::get<std::tuple_size<FeedbackType>::value - 1>(
        feedback).OutputParameter();

      // Transfer forward pass.
      Forward(feedbackLayerOutput, transfer);
      SaveActivations(transferActivations, transfer,
          transferActivationsCounter);
    }

    // Update the input for the next run
    locatorInput.push_back(new arma::cube(
        std::get<std::tuple_size<TransferType>::value - 1>(
        transfer).OutputParameter().memptr(), locatorInput.back().n_rows,
        locatorInput.back().n_cols, locatorInput.back().n_slices));
  }

  // Classifier forward pass.
  Forward(locatorInput.back().slice(0), classifier);

  // Reward predictor forward pass.
  Forward(std::get<std::tuple_size<ClassifierType>::value - 1>(
      classifier).OutputParameter(), rewardPredictor);

  double performanceError = negativeLogLikelihoodFunction.Forward(
      std::get<std::tuple_size<ClassifierType>::value - 1>(
      classifier).OutputParameter(), target);

  // Create the input for the vRClassRewardFunction function.
  // For which we use the output from the classifier and the rewardPredictor.
  rewardInput(0, 0) = std::get<std::tuple_size<ClassifierType>::value - 1>(
      classifier).OutputParameter();
  rewardInput(1, 0) = std::get<std::tuple_size<RewardPredictorType>::value - 1>(
      rewardPredictor).OutputParameter();

  performanceError += vRClassRewardFunction.Forward(rewardInput, target);

  return performanceError;
}

template<
  typename LocatorType,
  typename LocationSensorType,
  typename GlimpseSensorType,
  typename GlimpseType,
  typename StartType,
  typename FeedbackType,
  typename TransferType,
  typename ClassifierType,
  typename RewardPredictorType,
  typename InitializationRuleType,
  typename MatType
>
void RecurrentNeuralAttention<
  LocatorType,
  LocationSensorType,
  GlimpseSensorType,
  GlimpseType,
  StartType,
  FeedbackType,
  TransferType,
  ClassifierType,
  RewardPredictorType,
  InitializationRuleType,
  MatType
>::Gradient(const arma::mat& /* unused */,
            const size_t i,
            arma::mat& gradient)
{
  Evaluate(parameter, i, false);

  // Reset the gradient.
  if (gradient.is_empty())
  {
    gradient = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  // Reset the recurrent gradient.
  if (recurrentGradient.is_empty())
  {
    recurrentGradient = arma::zeros<arma::mat>(parameter.n_rows,
        parameter.n_cols);

    actionError = arma::zeros<arma::mat>(
        std::get<std::tuple_size<LocatorType>::value - 1>(
        locator).OutputParameter().n_rows,
        std::get<std::tuple_size<LocatorType>::value - 1>(
        locator).OutputParameter().n_cols);
  }
  else
  {
    recurrentGradient.zeros();
  }

  // Set the recurrent gradient.
  NetworkGradients(recurrentGradient, this->locator);
  NetworkGradients(recurrentGradient, this->locationSensor, locatorSize);
  NetworkGradients(recurrentGradient, this->glimpseSensor, locatorSize +
      locationSensorSize);
  NetworkGradients(recurrentGradient, this->glimpse, locatorSize +
      locationSensorSize + glimpseSensorSize);
  NetworkGradients(recurrentGradient, this->feedback, locatorSize +
      locationSensorSize + glimpseSensorSize + glimpseSize);
  NetworkGradients(recurrentGradient, this->transfer, locatorSize +
      locationSensorSize + glimpseSensorSize + glimpseSize + feedbackSize);

  // Set the gradient.
  NetworkGradients(gradient, this->classifier, locatorSize + locationSensorSize
      + glimpseSensorSize + glimpseSize + feedbackSize + transferSize);
  NetworkGradients(gradient, this->rewardPredictor, locatorSize +
      locationSensorSize + glimpseSensorSize + glimpseSize + feedbackSize +
      transferSize + classifierSize);
  NetworkGradients(gradient, this->start, locatorSize + locationSensorSize +
      glimpseSensorSize + glimpseSize + feedbackSize + transferSize +
      classifierSize + rewardPredictorSize);

  // Negative log likelihood backward pass.
  negativeLogLikelihoodFunction.Backward(std::get<std::tuple_size<
      ClassifierType>::value - 1>(classifier).OutputParameter(), target,
      negativeLogLikelihoodFunction.OutputParameter());

  const double reward = vRClassRewardFunction.Backward(rewardInput, target,
      vRClassRewardFunction.OutputParameter());

  // Propogate reward through all modules.
  ResetReward(reward, locator);
  ResetReward(reward, locationSensor);
  ResetReward(reward, glimpseSensor);
  ResetReward(reward, glimpse);
  ResetReward(reward, classifier);

  // RewardPredictor backward pass.
  Backward(vRClassRewardFunction.OutputParameter()(1, 0), rewardPredictor);

  arma::mat classifierError =
    negativeLogLikelihoodFunction.OutputParameter() +
    vRClassRewardFunction.OutputParameter()(0, 0) +
    std::get<0>(rewardPredictor).Delta();

  // Classifier backward pass.
  Backward(classifierError, classifier);

  // Set the initial recurrent error for the first backward step.
  arma::mat recurrentError = std::get<0>(classifier).Delta();

  for (step = nStep - 1; nStep >= 0; step--)
  {
    // Load the locator activations.
    LoadActivations(locatorInput[step], locatorActivations,
        locatorActivationsCounter, locator);

    // Load the location sensor activations.
    LoadActivations(std::get<std::tuple_size<LocatorType>::value - 1>(
        locator).OutputParameter(), locationSensorActivations,
        locationSensorActivationsCounter, locationSensor);

    // Load the glimpse sensor activations.
    LoadActivations(input, glimpseSensorMatActivations,
        glimpseSensorCubeActivations, glimpseSensorMatCounter,
        glimpseSensorCubeCounter, glimpseSensor);

    // Concat the parameter activation from the location and glimpse sensor.
    arma::mat concatLayerOutput = arma::join_cols(
        std::get<std::tuple_size<LocationSensorType>::value - 1>(
        locationSensor).OutputParameter(),
        std::get<std::tuple_size<GlimpseSensorType>::value - 1>(
        glimpseSensor).OutputParameter());

    // Load the glimpse activations.
    LoadActivations(concatLayerOutput, glimpseActivations,
        glimpseActivationsCounter, glimpse);


    if (step == 0)
    {
      // Load the transfer activations.
     LoadActivations(std::get<std::tuple_size<StartType>::value - 1>(
          start).OutputParameter(), transferActivations,
          transferActivationsCounter, transfer);
    }
    else
    {
      // Load the feedback activations.
      LoadActivations(std::get<std::tuple_size<TransferType>::value - 1>(
          transfer).OutputParameter(), feedbackActivations,
          feedbackActivationsCounter, feedback);

      arma::mat feedbackLayerOutput =
        std::get<std::tuple_size<GlimpseType>::value - 1>(
        glimpse).OutputParameter() +
        std::get<std::tuple_size<FeedbackType>::value - 1>(
        feedback).OutputParameter();

      // Load the transfer activations.
      LoadActivations(feedbackLayerOutput, transferActivations,
          transferActivationsCounter, transfer);
    }

    // Set the location parameter for all layer that implement a Location
    // function e.g. GlimpseLayer.
    ResetLocation(location[step], glimpseSensor);

    // Locator backward pass.
    Backward(actionError, locator);

    // Transfer backward pass.
    Backward(recurrentError, transfer);

    // glimpse network
    Backward(std::get<0>(transfer).Delta(), glimpse);

    // Split up the error of the concat layer.
    arma::mat locationSensorError = std::get<0>(glimpse).Delta().submat(
        0, 0, std::get<0>(glimpse).Delta().n_elem / 2 - 1, 0);
    arma::mat glimpseSensorError = std::get<0>(glimpse).Delta().submat(
        std::get<0>(glimpse).Delta().n_elem / 2, 0,
        std::get<0>(glimpse).Delta().n_elem - 1, 0);

    // Location sensor backward pass.
    Backward(locationSensorError, locationSensor);

    // Glimpse sensor backward pass.
    Backward(glimpseSensorError, glimpseSensor);

    if (step != 0)
    {
      // Feedback backward pass.
      Backward(std::get<0>(transfer).Delta(), feedback);
    }

    // Update the recurrent network gradients.
    UpdateGradients(std::get<0>(locationSensor).Delta(), locator);
    UpdateGradients(std::get<0>(transfer).Delta(), glimpse);
    UpdateGradients(std::get<0>(transfer).Delta(), locationSensor);
    UpdateGradients(std::get<0>(transfer).Delta(), glimpseSensor);

    // Feedback module.
    if (step != 0)
    {
      UpdateGradients(feedbackActivationsInput[step - 1],
          std::get<0>(transfer).Delta(), feedback);
    }
    else
    {
      // Set the feedback gradient to zero.
      recurrentGradient.submat(locatorSize + locationSensorSize +
          glimpseSensorSize + glimpseSize, 0, locatorSize + locationSensorSize +
          glimpseSensorSize + glimpseSize + feedbackSize - 1, 0).zeros();

      UpdateGradients(std::get<0>(transfer).Delta(), start);
    }

    // Update the overall recurrent gradient.
    gradient += recurrentGradient;

    if (step != 0)
    {
      // Update the recurrent error for the next backward step.
      recurrentError = std::get<0>(locator).Delta() +
          std::get<0>(feedback).Delta();
    }
    else
    {
      break;
    }
  }

  // Reward predictor gradient update.
  UpdateGradients(vRClassRewardFunction.OutputParameter()(1, 0),
      rewardPredictor);

  // Classifier gradient update.
  UpdateGradients(std::get<1>(classifier).Delta(), classifier);
}

template<
  typename LocatorType,
  typename LocationSensorType,
  typename GlimpseSensorType,
  typename GlimpseType,
  typename StartType,
  typename FeedbackType,
  typename TransferType,
  typename ClassifierType,
  typename RewardPredictorType,
  typename InitializationRuleType,
  typename MatType
>
const arma::mat& RecurrentNeuralAttention<
  LocatorType,
  LocationSensorType,
  GlimpseSensorType,
  GlimpseType,
  StartType,
  FeedbackType,
  TransferType,
  ClassifierType,
  RewardPredictorType,
  InitializationRuleType,
  MatType
>::Location()
{
  if (!location.empty())
  {
    evaluationLocation = arma::mat(location[0].n_elem, location.size());

    for (size_t i = 0; i < location.size(); i++)
    {
      evaluationLocation.col(i) = arma::vectorise(location[i]);
    }
  }

  return evaluationLocation;
}

template<
  typename LocatorType,
  typename LocationSensorType,
  typename GlimpseSensorType,
  typename GlimpseType,
  typename StartType,
  typename FeedbackType,
  typename TransferType,
  typename ClassifierType,
  typename RewardPredictorType,
  typename InitializationRuleType,
  typename MatType
>
template<typename Archive>
void RecurrentNeuralAttention<
  LocatorType,
  LocationSensorType,
  GlimpseSensorType,
  GlimpseType,
  StartType,
  FeedbackType,
  TransferType,
  ClassifierType,
  RewardPredictorType,
  InitializationRuleType,
  MatType
>::Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(parameter, "parameter");
  ar & data::CreateNVP(inputSize, "inputSize");
  ar & data::CreateNVP(nStep, "nStep");

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    // Set the netork size.
    locatorSize = NetworkSize(this->locator);
    locationSensorSize = NetworkSize(this->locationSensor);
    glimpseSensorSize = NetworkSize(this->glimpseSensor);
    glimpseSize = NetworkSize(this->glimpse);
    feedbackSize = NetworkSize(this->feedback);
    transferSize = NetworkSize(this->transfer);
    classifierSize = NetworkSize(this->classifier);
    rewardPredictorSize = NetworkSize(this->rewardPredictor);
    startSize = NetworkSize(this->start);

    // Set the network weights.
    NetworkWeights(parameter, this->locator);
    NetworkWeights(parameter, this->locationSensor, locatorSize);
    NetworkWeights(parameter, this->glimpseSensor, locatorSize +
        locationSensorSize);
    NetworkWeights(parameter, this->glimpse, locatorSize + locationSensorSize +
        glimpseSensorSize);
    NetworkWeights(parameter, this->feedback, locatorSize + locationSensorSize +
        glimpseSensorSize + glimpseSize);
    NetworkWeights(parameter, this->transfer, locatorSize + locationSensorSize +
        glimpseSensorSize + glimpseSize + feedbackSize);
    NetworkWeights(parameter, this->classifier, locatorSize + locationSensorSize
        + glimpseSensorSize + glimpseSize + feedbackSize + transferSize);
    NetworkWeights(parameter, this->rewardPredictor, locatorSize +
        locationSensorSize + glimpseSensorSize + glimpseSize + feedbackSize +
        transferSize + classifierSize);
    NetworkWeights(parameter, this->start, locatorSize + locationSensorSize +
        glimpseSensorSize + glimpseSize + feedbackSize + transferSize +
        classifierSize + rewardPredictorSize);
  }
}

} // namespace ann
} // namespace mlpack

#endif
