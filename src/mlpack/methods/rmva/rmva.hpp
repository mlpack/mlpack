/**
 * @file rmva.hpp
 * @author Marcus Edel
 *
 * Definition of the RecurrentNeuralAttention class, which implements the
 * Recurrent Model for Visual Attention.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_RMVA_RMVA_HPP
#define __MLPACK_METHODS_RMVA_RMVA_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/network_util.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/performance_functions/cee_function.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/layer/negative_log_likelihood_layer.hpp>
#include <mlpack/methods/ann/layer/vr_class_reward_layer.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

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
 * @tparam LocatorType Type of locator network.
 * @tparam LocationSensorType Type of location sensor network.
 * @tparam GlimpseSensorType Type of glimpse sensor network.
 * @tparam GlimpseType Type of glimpse network.
 * @tparam StartType Type of start network.
 * @tparam FeedbackType Type of feedback network.
 * @tparam TransferType Type of transfer network.
 * @tparam ClassifierType Type of classifier network.
 * @tparam RewardPredictorType Type of reward predictor network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam MatType Matrix type (arma::mat or arma::sp_mat).
 */
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
  typename InitializationRuleType = RandomInitialization,
  typename MatType = arma::mat
>
class RecurrentNeuralAttention
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = RecurrentNeuralAttention<
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
      MatType>;

  /**
   * Construct the RecurrentNeuralAttention object, which will construct the
   * recurrent model for visual attentionh using the specified networks.
   *
   * @param locator The locator network.
   * @param locationSensor The location sensor network.
   * @param glimpseSensor The glimpse sensor network.
   * @param glimpse The glimpse network.
   * @param start The start network.
   * @param feedback The feedback network.
   * @param transfer The transfer network.
   * @param classifier The classifier network.
   * @param rewardPredictor The reward predictor network.
   * @param nStep Number of steps for the back-propagate through time.
   * @param initializeRule Rule used to initialize the weight matrix.
   */
  template<typename TypeLocator,
           typename TypeLocationSensor,
           typename TypeGlimpseSensor,
           typename TypeGlimpse,
           typename TypeStart,
           typename TypeFeedback,
           typename TypeTransfer,
           typename TypeClassifier,
           typename TypeRewardPredictor>
  RecurrentNeuralAttention(TypeLocator&& locator,
                           TypeLocationSensor&& locationSensor,
                           TypeGlimpseSensor&& glimpseSensor,
                           TypeGlimpse&& glimpse,
                           TypeStart&& start,
                           TypeFeedback&& feedback,
                           TypeTransfer&& transfer,
                           TypeClassifier&& classifier,
                           TypeRewardPredictor&& rewardPredictor,
                           const size_t nStep,
                           InitializationRuleType initializeRule =
                              InitializationRuleType());
  /**
   * Train the network on the given input data using the given optimizer.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   */
  template<
      template<typename> class OptimizerType = mlpack::optimization::RMSprop
  >
  void Train(const arma::mat& predictors,
             const arma::mat& responses,
             OptimizerType<NetworkType>& optimizer);

  /**
   * Predict the responses to a given set of predictors. The responses will
   * reflect the output of the given output layer as returned by the
   * OutputClass() function.
   *
   * @param predictors Input predictors.
   * @param responses Matrix to put output predictions of responses into.
   */
  void Predict(arma::mat& predictors, arma::mat& responses);

  /**
   * Evaluate the network with the given parameters. This function is usually
   * called by the optimizer to train the model.
   *
   * @param parameters Matrix model parameters.
   * @param i Index of point to use for objective function evaluation.
   * @param deterministic Whether or not to train or test the model. Note some
   * layer act differently in training or testing mode.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t i,
                  const bool deterministic = true);

  /**
   * Evaluate the gradient of the network with the given parameters, and with
   * respect to only one point in the dataset. This is useful for
   * optimizers such as SGD, which require a separable objective function.
   *
   * @param parameters Matrix of the model parameters to be optimized.
   * @param i Index of points to use for objective function gradient evaluation.
   * @param gradient Matrix to output gradient into.
   */
  void Gradient(const arma::mat& parameters,
                const size_t i,
                arma::mat& gradient);

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }

  //! Return the number of steps to back-propagate through time.
  const size_t& Rho() const { return nStep; }
  //! Modify the number of steps to back-propagate through time.
  size_t& Rho() { return nStep; }

  //! Return the current location.
  const arma::mat& Location();

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  /*
   * Predict the response of the given input matrix.
   */
  template <typename InputType, typename OutputType>
  void SinglePredict(const InputType& input, OutputType& output)
  {
    // Get the locator input size.
    if (!inputSize)
    {
      inputSize = NetworkInputSize(locator);
    }

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

    // Sample an initial starting actions by forwarding zeros through the
    // locator.
    locatorInput.push_back(new arma::cube(arma::zeros<arma::cube>(inputSize, 1,
        input.n_slices)));

    // Forward pass throught the recurrent network.
    for (step = 0; step < nStep; step++)
    {
      // Locator forward pass.
      Forward(locatorInput.back(), locator);

      // Location sensor forward pass.
      Forward(std::get<std::tuple_size<LocatorType>::value - 1>(
          locator).OutputParameter(), locationSensor);

      // Set the location parameter for all layer that implement a Location
      // function e.g. GlimpseLayer.
      ResetLocation(std::get<std::tuple_size<LocatorType>::value - 1>(
          locator).OutputParameter(), glimpseSensor);

      // Glimpse sensor forward pass.
      Forward(input, glimpseSensor);

      // Concat the parameter activation from the location sensor and
      // glimpse sensor.
      arma::mat concatLayerOutput = arma::join_cols(
          std::get<std::tuple_size<LocationSensorType>::value - 1>(
          locationSensor).OutputParameter(),
          std::get<std::tuple_size<GlimpseSensorType>::value - 1>(
          glimpseSensor).OutputParameter());

      // Glimpse forward pass.
      Forward(concatLayerOutput, glimpse);

      if (step == 0)
      {
        // Start forward pass.
        Forward(std::get<std::tuple_size<GlimpseType>::value - 1>(
            glimpse).OutputParameter(), start);

        // Transfer forward pass.
        Forward(std::get<std::tuple_size<StartType>::value - 1>(
            start).OutputParameter(), transfer);
      }
      else
      {
        // Feedback forward pass.
        Forward(std::get<std::tuple_size<TransferType>::value - 1>(
            transfer).OutputParameter(), feedback);

        arma::mat feedbackLayerOutput =
          std::get<std::tuple_size<GlimpseType>::value - 1>(
          glimpse).OutputParameter() +
          std::get<std::tuple_size<FeedbackType>::value - 1>(
          feedback).OutputParameter();

        // Transfer forward pass.
        Forward(feedbackLayerOutput, transfer);
      }

      // Update the input for the next run
      locatorInput.push_back(new arma::cube(
          std::get<std::tuple_size<TransferType>::value - 1>(
          transfer).OutputParameter().memptr(), locatorInput.back().n_rows,
          locatorInput.back().n_cols, locatorInput.back().n_slices));
    }

    // Classifier forward pass.
    Forward(locatorInput.back().slice(0), classifier);

    output = std::get<std::tuple_size<ClassifierType>::value - 1>(
        classifier).OutputParameter();
  }

  /**
   * Update the layer reward for all layer that implement the Rewards function.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetReward(const double reward, std::tuple<Tp...>& network)
  {
    SetReward(reward, std::get<I>(network));
    ResetReward<I + 1, Tp...>(reward, network);
  }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ResetReward(const double /* reward */, std::tuple<Tp...>& /* network */)
  {
  }

  template<typename T>
  typename std::enable_if<
      HasRewardCheck<T, double&(T::*)()>::value, void>::type
  SetReward(const double reward, T& layer)
  {
    layer.Reward() = reward;
  }

  template<typename T>
  typename std::enable_if<
      !HasRewardCheck<T, double&(T::*)()>::value, void>::type
  SetReward(const double /* reward */, T& /* layer */)
  {
    /* Nothing to do here */
  }

  /**
   * Reset the network by clearing the delta and by setting the layer status.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& /* network */) { /* Nothing to do here */ }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetParameter(std::tuple<Tp...>& network)
  {
    ResetDeterministic(std::get<I>(network));
    std::get<I>(network).Delta().zeros();

    ResetParameter<I + 1, Tp...>(network);
  }

  template<typename T>
  typename std::enable_if<
      HasDeterministicCheck<T, bool&(T::*)(void)>::value, void>::type
  ResetDeterministic(T& layer)
  {
    layer.Deterministic() = deterministic;
  }

  template<typename T>
  typename std::enable_if<
      !HasDeterministicCheck<T, bool&(T::*)(void)>::value, void>::type
  ResetDeterministic(T& /* layer */) { /* Nothing to do here */ }

  /**
   * Reset the location by updating the location for all layer that implement
   * the Location function.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ResetLocation(const arma::mat& /* location */,
                std::tuple<Tp...>& /* network */)
  {
    // Nothing to do here.
  }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ResetLocation(const arma::mat& location, std::tuple<Tp...>& network)
  {
    SetLocation(std::get<I>(network), location);
    ResetLocation<I + 1, Tp...>(location, network);
  }

  template<typename T>
  typename std::enable_if<
      HasLocationCheck<T, void(T::*)(const arma::mat&)>::value, void>::type
  SetLocation(T& layer, const arma::mat& location)
  {
    layer.Location(location);
  }

  template<typename T>
  typename std::enable_if<
      !HasLocationCheck<T, void(T::*)(const arma::mat&)>::value, void>::type
  SetLocation(T& /* layer */, const arma::mat& /* location */)
  {
    // Nothing to do here.
  }

  /**
   * Save the network layer activations.
   */
  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  SaveActivations(boost::ptr_vector<MatType>& activations,
                  std::tuple<Tp...>& network,
                  size_t& activationCounter)
  {
    Save(I, activations, std::get<I>(network),
        std::get<I>(network).InputParameter());

    activationCounter++;
    SaveActivations<I + 1, Tp...>(activations, network, activationCounter);
  }

  template<size_t I = 0, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  SaveActivations(boost::ptr_vector<MatType>& /* activations */,
                  std::tuple<Tp...>& /* network */,
                  size_t& /* activationCounter */)
  {
    // Nothing to do here.
  }

  /**
   * Distinguish between recurrent layer and non-recurrent layer when storing
   * the activations.
   */
  template<typename T, typename P>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Save(const size_t /* layerNumber */,
       boost::ptr_vector<MatType>& activations,
       T& layer,
       P& /* unused */)
  {
    activations.push_back(new MatType(layer.RecurrentParameter()));
  }

  template<typename T, typename P>
  typename std::enable_if<
      !HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Save(const size_t /* layerNumber */,
       boost::ptr_vector<MatType>& activations,
       T& layer,
       P& /* unused */)
  {
    activations.push_back(new MatType(layer.OutputParameter()));
  }

  template<size_t I = 0, typename DataTypeA, typename DataTypeB, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  SaveActivations(boost::ptr_vector<DataTypeA>& activationsA,
                  boost::ptr_vector<DataTypeB>& activationsB,
                  size_t& dataTypeACounter,
                  size_t& dataTypeBCounter,
                  std::tuple<Tp...>& network)
  {
    Save(activationsA, activationsB, dataTypeACounter, dataTypeBCounter,
        std::get<I>(network), std::get<I>(network).OutputParameter());

    SaveActivations<I + 1, DataTypeA, DataTypeB, Tp...>(
        activationsA, activationsB, dataTypeACounter, dataTypeBCounter,
        network);
  }

  template<size_t I = 0, typename DataTypeA, typename DataTypeB, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  SaveActivations(boost::ptr_vector<DataTypeA>& /* activationsA */,
                  boost::ptr_vector<DataTypeB>& /* activationsB */,
                  size_t& /* dataTypeACounter */,
                  size_t& /* dataTypeBCounter */,
                  std::tuple<Tp...>& /* network */)
  {
    // Nothing to do here.
  }

  template<typename T, typename DataTypeA, typename DataTypeB>
  void Save(boost::ptr_vector<DataTypeA>& activationsA,
        boost::ptr_vector<DataTypeB>& /* activationsB */,
       size_t& dataTypeACounter,
       size_t& /* dataTypeBCounter */,
       T& layer,
       DataTypeA& /* unused */)
  {
    activationsA.push_back(new DataTypeA(layer.OutputParameter()));
    dataTypeACounter++;
  }

  template<typename T, typename DataTypeA, typename DataTypeB>
  void Save(boost::ptr_vector<DataTypeA>& /* activationsA */,
            boost::ptr_vector<DataTypeB>& activationsB,
            size_t& /* dataTypeACounter */,
            size_t& dataTypeBCounter,
            T& layer,
            DataTypeB& /* unused */)
  {
    activationsB.push_back(new DataTypeB(layer.OutputParameter()));
    dataTypeBCounter++;
  }

  /**
   * Load the network layer activations.
   */
  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  LoadActivations(DataType& input,
                  boost::ptr_vector<MatType>& /* activations */,
                  size_t& /* activationCounter */,
                  std::tuple<Tp...>& network)
  {
    std::get<0>(network).InputParameter() = input;
    LinkParameter(network);
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  LoadActivations(DataType& input,
                  boost::ptr_vector<MatType>& activations,
                  size_t& activationCounter,
                  std::tuple<Tp...>& network)
  {
    Load(--activationCounter, activations,
        std::get<sizeof...(Tp) - I - 1>(network),
        std::get<I>(network).InputParameter());

    LoadActivations<I + 1, DataType, Tp...>(input, activations,
        activationCounter, network);
  }

  /**
   * Distinguish between recurrent layer and non-recurrent layer when storing
   * the activations.
   */
  template<typename T, typename P>
  typename std::enable_if<
      HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Load(const size_t layerNumber,
       boost::ptr_vector<MatType>& activations,
       T& layer,
       P& /* output */)
  {
    layer.RecurrentParameter() = activations[layerNumber];
  }

  template<typename T, typename P>
  typename std::enable_if<
      !HasRecurrentParameterCheck<T, P&(T::*)()>::value, void>::type
  Load(const size_t layerNumber,
       boost::ptr_vector<MatType>& activations,
       T& layer,
       P& /* output */)
  {
    layer.OutputParameter() = activations[layerNumber];
  }

  template<size_t I = 0,
           typename DataType,
           typename DataTypeA,
           typename DataTypeB,
           typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  LoadActivations(DataType& input,
                  boost::ptr_vector<DataTypeA>& activationsA,
                  boost::ptr_vector<DataTypeB>& activationsB,
                  size_t& dataTypeACounter,
                  size_t& dataTypeBCounter,
                  std::tuple<Tp...>& network)
  {
    Load(activationsA,
         activationsB,
         dataTypeACounter,
         dataTypeBCounter,
         std::get<sizeof...(Tp) - I - 1>(network),
         std::get<sizeof...(Tp) - I - 1>(network).OutputParameter());

    LoadActivations<I + 1, DataType, DataTypeA, DataTypeB, Tp...>(
        input, activationsA, activationsB, dataTypeACounter, dataTypeBCounter,
        network);
  }

  template<size_t I = 0,
           typename DataType,
           typename DataTypeA,
           typename DataTypeB,
           typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  LoadActivations(DataType& input,
                  boost::ptr_vector<DataTypeA>& /* activationsA */,
                  boost::ptr_vector<DataTypeB>& /* activationsB */,
                  size_t& /* dataTypeACounter */,
                  size_t& /* dataTypeBCounter */,
                  std::tuple<Tp...>& network)
  {
    std::get<0>(network).InputParameter() = input;
    LinkParameter(network);
  }

  template<typename T, typename DataTypeA, typename DataTypeB>
  void Load(boost::ptr_vector<DataTypeA>& activationsA,
            boost::ptr_vector<DataTypeB>& /* activationsB */,
            size_t& dataTypeACounter,
            size_t& /* dataTypeBCounter */,
            T& layer,
            DataTypeA& /* output */)
  {
    layer.OutputParameter() = activationsA[--dataTypeACounter];
  }

  template<typename T, typename DataTypeA, typename DataTypeB>
  void Load(boost::ptr_vector<DataTypeA>& /* activationsA */,
            boost::ptr_vector<DataTypeB>& activationsB,
            size_t& /* dataTypeACounter */,
            size_t& dataTypeBCounter,
            T& layer,
            DataTypeB& /* output */)
  {
    layer.OutputParameter() = activationsB[--dataTypeBCounter];
  }

  /**
   * Run a single iteration of the feed forward algorithm, using the given
   * input and target vector, store the calculated error into the error
   * vector.
   */
  template<size_t I = 0, typename DataType, typename... Tp>
  void Forward(const DataType& input, std::tuple<Tp...>& t)
  {
    std::get<I>(t).InputParameter() = input;
    std::get<I>(t).Forward(std::get<I>(t).InputParameter(),
        std::get<I>(t).OutputParameter());

    ForwardTail<I + 1, Tp...>(t);
  }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& network)
  {
    LinkParameter(network);
  }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& t)
  {
    std::get<I>(t).Forward(std::get<I - 1>(t).OutputParameter(),
        std::get<I>(t).OutputParameter());

    ForwardTail<I + 1, Tp...>(t);
  }

  /**
   * Run a single iteration of the backward algorithm, using the given
   * input and target vector, store the calculated error into the error
   * vector.
   */
  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<sizeof...(Tp) == 1, void>::type
  Backward(const DataType& error, std::tuple<Tp ...>& t)
  {
    std::get<sizeof...(Tp) - I>(t).Backward(
      std::get<sizeof...(Tp) - I>(t).OutputParameter(), error,
      std::get<sizeof...(Tp) - I>(t).Delta());
  }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  Backward(const DataType& error, std::tuple<Tp ...>& t)
  {
    std::get<sizeof...(Tp) - I>(t).Backward(
        std::get<sizeof...(Tp) - I>(t).OutputParameter(), error,
        std::get<sizeof...(Tp) - I>(t).Delta());

    BackwardTail<I + 1, DataType, Tp...>(error, t);
  }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I == (sizeof...(Tp)), void>::type
  BackwardTail(const DataType& /* error */, std::tuple<Tp...>& t)
  {
    std::get<sizeof...(Tp) - I>(t).Backward(
        std::get<sizeof...(Tp) - I>(t).OutputParameter(),
        std::get<sizeof...(Tp) - I + 1>(t).Delta(),
        std::get<sizeof...(Tp) - I>(t).Delta());
  }

  template<size_t I = 1, typename DataType, typename... Tp>
  typename std::enable_if<I < (sizeof...(Tp)), void>::type
  BackwardTail(const DataType& error, std::tuple<Tp...>& t)
  {
    std::get<sizeof...(Tp) - I>(t).Backward(
        std::get<sizeof...(Tp) - I>(t).OutputParameter(),
        std::get<sizeof...(Tp) - I + 1>(t).Delta(),
        std::get<sizeof...(Tp) - I>(t).Delta());

    BackwardTail<I + 1, DataType, Tp...>(error, t);
  }

  /**
   * Link the calculated activation with the correct layer.
   */
  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  LinkParameter(std::tuple<Tp ...>& /* network */) { /* Nothing to do here */ }

  template<size_t I = 1, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  LinkParameter(std::tuple<Tp...>& network)
  {
    if (!LayerTraits<typename std::remove_reference<
        decltype(std::get<I>(network))>::type>::IsBiasLayer)
    {
      std::get<I>(network).InputParameter() = std::get<I - 1>(
          network).OutputParameter();
    }

    LinkParameter<I + 1, Tp...>(network);
  }

  /**
   * Iterate through all layer modules and update the the gradient using the
   * layer defined optimizer.
   */
  template<typename InputType, typename ErrorType, typename... Tp>
  void UpdateGradients(const InputType& input,
                       const ErrorType& error,
                       std::tuple<Tp...>& network)
  {
     Update(std::get<0>(network),
           input,
           std::get<1>(network).Delta(),
           std::get<1>(network).OutputParameter());

     UpdateGradients<1, ErrorType, Tp...>(error, network);
  }

  template<size_t I = 0, typename ErrorType, typename... Tp>
  typename std::enable_if<I < (sizeof...(Tp) - 1), void>::type
  UpdateGradients(const ErrorType& error, std::tuple<Tp...>& network)
  {
    Update(std::get<I>(network),
           std::get<I>(network).InputParameter(),
           std::get<I + 1>(network).Delta(),
           std::get<I>(network).OutputParameter());

    UpdateGradients<I + 1, ErrorType, Tp...>(error, network);
  }

  template<size_t I = 0, typename ErrorType, typename... Tp>
  typename std::enable_if<I == (sizeof...(Tp) - 1), void>::type
  UpdateGradients(const ErrorType& error, std::tuple<Tp...>& network)
  {
    Update(std::get<I>(network),
       std::get<I>(network).InputParameter(),
       error,
       std::get<I>(network).OutputParameter());
  }

  template<typename LayerType,
           typename InputType,
           typename ErrorType,
           typename GradientType>
  typename std::enable_if<
      HasGradientCheck<LayerType,
          void(LayerType::*)(const InputType&,
                             const ErrorType&,
                             GradientType&)>::value, void>::type
  Update(LayerType& layer,
         const InputType& input,
         const ErrorType& error,
         GradientType& /* gradient */)
  {
    layer.Gradient(input, error, layer.Gradient());
  }

  template<typename LayerType,
           typename InputType,
           typename ErrorType,
           typename GradientType>
  typename std::enable_if<
      !HasGradientCheck<LayerType,
          void(LayerType::*)(const InputType&,
                             const ErrorType&,
                             GradientType&)>::value, void>::type
  Update(LayerType& /* layer */,
         const InputType& /* input */,
         const ErrorType& /* error */,
         GradientType& /* gradient */)
  {
    // Nothing to do here
  }

  //! The locator network.
  LocatorType locator;

  //! The location sensor network.
  LocationSensorType locationSensor;

  //! The glimpse sensor network.
  GlimpseSensorType glimpseSensor;

  //! The glimpse network.
  GlimpseType glimpse;

  //! The start network.
  StartType start;

  //! The feedback network.
  FeedbackType feedback;

  //! The transfer network.
  TransferType transfer;

  //! The classifier network.
  ClassifierType classifier;

  //! The reward predictor network.
  RewardPredictorType rewardPredictor;

  //! The number of steps for the back-propagate through time.
  size_t nStep;

  //! Locally stored network input size.
  size_t inputSize;

  //! The current evaluation mode (training or testing).
  bool deterministic;

  //! The index of the current step.
  size_t step;

  //! The activation storage we are using to perform the feed backward pass for
  //! the glimpse network.
  boost::ptr_vector<arma::mat> glimpseActivations;

  //! The activation storage we are using to perform the feed backward pass for
  //! the locator network.
  boost::ptr_vector<arma::mat> locatorActivations;

  //! The activation storage we are using to perform the feed backward pass for
  //! the feedback network.
  boost::ptr_vector<arma::mat> feedbackActivations;

  //! The activation storage we are using to save the feedback network input.
  boost::ptr_vector<arma::mat> feedbackActivationsInput;

  //! The activation storage we are using to perform the feed backward pass for
  //! the transfer network.
  boost::ptr_vector<arma::mat> transferActivations;

  //! The activation storage we are using to perform the feed backward pass for
  //! the location sensor network.
  boost::ptr_vector<arma::mat> locationSensorActivations;

  //! The activation storage we are using to perform the feed backward pass for
  //! the glimpse sensor network.
  boost::ptr_vector<arma::mat> glimpseSensorMatActivations;
  boost::ptr_vector<arma::cube> glimpseSensorCubeActivations;

  //! The activation storage we are using to perform the feed backward pass for
  //! the locator input.
  boost::ptr_vector<arma::cube> locatorInput;

  //! The storage we are using to save the location.
  boost::ptr_vector<arma::mat> location;

  //! The current number of activations in the glimpse sensor network.
  size_t glimpseSensorMatCounter;
  size_t glimpseSensorCubeCounter;

  //! The current number of activations in the glimpse network.
  size_t glimpseActivationsCounter;

  //! The current number of activations in the glimpse start network.
  size_t startActivationsCounter;

  //! The current number of activations in the feedback network.
  size_t feedbackActivationsCounter;

  //! The current number of activations in the transfer network.
  size_t transferActivationsCounter;

  //! The current number of activations in the locator network.
  size_t locatorActivationsCounter;

  //! The current number of activations in the location sensor network.
  size_t locationSensorActivationsCounter;

  //! The current number of activations in the glimpse sensor network.
  size_t glimpseSensorMatActivationsCounter;
  size_t glimpseSensorCubeActivationsCounter;

  //! The current number of location for the location storage.
  size_t locationCounter;

  //! Matrix of (trained) parameters.
  arma::mat parameter;

  //! The matrix of data points (predictors).
  arma::mat predictors;

  //! The matrix of responses to the input data points.
  arma::mat responses;

  //! The number of separable functions (the number of predictor points).
  size_t numFunctions;

  //! Storage the merge the reward input.
  arma::field<arma::mat> rewardInput;

  //! The current input.
  arma::cube input;

  //! The current target.
  arma::mat target;

  //! Locally stored performance functions.
  NegativeLogLikelihoodLayer<> negativeLogLikelihoodFunction;
  VRClassRewardLayer<> vRClassRewardFunction;

  //! Locally stored size of the locator network.
  size_t locatorSize;

  //! Locally stored size of the location sensor network.
  size_t locationSensorSize;

  //! Locally stored size of the glimpse sensor network.
  size_t glimpseSensorSize;

  //! Locally stored size of the glimpse network.
  size_t glimpseSize;

  //! Locally stored size of the start network.
  size_t startSize;

  //! Locally stored size of the feedback network.
  size_t feedbackSize;

  //! Locally stored size of the transfer network.
  size_t transferSize;

  //! Locally stored size of the classifier network.
  size_t classifierSize;

  //! Locally stored size of the reward predictor network.
  size_t rewardPredictorSize;

  //! Locally stored recurrent gradient.
  arma::mat recurrentGradient;

  //! Locally stored action error.
  arma::mat actionError;

  //! Locally stored current location.
  arma::mat evaluationLocation;
}; // class RecurrentNeuralAttention

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "rmva_impl.hpp"

#endif
