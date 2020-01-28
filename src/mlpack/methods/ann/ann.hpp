#ifndef MLPACK_METHODS_ANN_ANN_HPP
#define MLPACK_METHODS_ANN_ANN_HPP

#include <mlpack/prereqs.hpp>
#include <tuple>
#include <type_traits>

#include "ann_update.hpp"

namespace mlpack {
namespace ann {

template<typename InitializationRuleType, size_t NumLayers, typename OutputLayerType, typename... LayerTypes>
class ANN
{
 public:
  typedef ANN<InitializationRuleType, NumLayers, OutputLayerType, LayerTypes...> NetworkType;
  
  typedef void check;
  
  template<typename OptimizerType>
  using DefaultUpdatePolicy = ANNUpdate<OptimizerType, NetworkType>;
  
  template<typename OptimizerType, typename... Params>
  static DefaultUpdatePolicy<OptimizerType> GetDefaultUpdatePolicy(Params... params)
  {
    return DefaultUpdatePolicy<OptimizerType>(params...);
  }
 
  ANN(const LayerTypes& ... layers)
    : layers(std::make_tuple(layers..., OutputLayerType()))
  {}
  
  /**
   * Train the feedforward network on the given input data using the given
   * optimizer.
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
      template<typename, typename...> class OptimizerType =
          mlpack::optimization::RMSProp,
      typename... OptimizerTypeArgs
  >
  void Train(const arma::mat& predictors,
             const arma::mat& responses,
             OptimizerType<NetworkType, OptimizerTypeArgs...>& optimizer);
             
  /**
   * Evaluate the feedforward network with the given parameters. This function
   * is usually called by the optimizer to train the model.
   *
   * @param parameters Matrix model parameters.
   * @param i Index of point to use for objective function evaluation.
   * @param deterministic Whether or not to train or test the model. Note some
   *        layer act differently in training or testing mode.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t i,
                  const bool deterministic = true);
                  
  /**
   * Evaluate the gradient of the feedforward network with the given parameters,
   * and with respect to only one point in the dataset. This is useful for
   * optimizers such as SGD, which require a separable objective function.
   *
   * @param parameters Matrix of the model parameters to be optimized.
   * @param i Index of points to use for objective function gradient evaluation.
   * @param gradient Matrix to output gradient into.
   */
  void Gradient(const arma::mat& parameters,
                const size_t i,
                arma::mat& gradient);
                
  /**
   * Evaluate the gradient of the feedforward network with the given parameters,
   * and with respect to only one point in the dataset. This is useful for
   * optimizers such as SGD, which require a separable objective function.
   *
   * @param parameters Matrix of the model parameters to be optimized.
   * @param i Index of points to use for objective function gradient evaluation.
   * @param gradient Matrix to output gradient into.
   */
  template<typename UpdatePolicyType, typename... Params>
  void UpdateGradient(UpdatePolicyType& updatePolicy, Params... params);
                  
  void ResetData(const arma::mat& predictors, const arma::mat& responses);
  
  void ResetParameters();
  
  arma::mat& CurrentTarget()
  {
    return currentTarget;
  }
  
  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }
  
 private:
 
  double Forward(const arma::mat& input);
 
  /**
   * Prepare the network for the given data.
   * This function won't actually trigger training process.
   *
   * @param predictors Input data variables.
   * @param responses Outputs results from input data variables.
   */
  
  /** For Forward Pass */
  template<typename T, typename U>
  struct forward_call
  {
    static double call() { return 0; }
  };
  
  template<typename T, size_t LayerNum>
  struct forward_call<T, std::integral_constant<size_t, LayerNum>>
  {
    static double call(const arma::mat& input, std::tuple<LayerTypes..., OutputLayerType>& layers, NetworkType* network);
  };
  
  template<typename T>
  struct forward_call<T, std::integral_constant<size_t, NumLayers>>
  {
    static double call(const arma::mat& input, std::tuple<LayerTypes..., OutputLayerType>& layers, NetworkType* network);
  };
  /** Forward pass finish */
  
  /** For Backward pass */
  template<typename T, typename U>
  struct backward_call
  {
    static void call() { return 0; }
  };
  
  template<typename UpdatePolicy>
  struct backward_call<UpdatePolicy, std::integral_constant<size_t, NumLayers>>
  {
    static void call(UpdatePolicy& policy, arma::mat& error, arma::mat& currentTarget, arma::mat& currentInput, std::tuple<LayerTypes..., OutputLayerType>& layers);
  };
  
  template<typename UpdatePolicy>
  struct backward_call<UpdatePolicy, std::integral_constant<size_t, 0>>
  {
    static void call(UpdatePolicy& policy, arma::mat& error, arma::mat& currentTarget, arma::mat& currentInput, std::tuple<LayerTypes..., OutputLayerType>& layers);
  };
  
  template<typename UpdatePolicy, size_t LayerNum>
  struct backward_call<UpdatePolicy, std::integral_constant<size_t, LayerNum>>
  {
    static void call(UpdatePolicy& policy, arma::mat& error, arma::mat& currentTarget, arma::mat& currentInput, std::tuple<LayerTypes..., OutputLayerType>& layers);
  };
  /** Backward pass finish */
  
  template<class>
  struct sfinae_true : std::true_type{};

  template<class T>
  static auto has_parameters(int) -> sfinae_true<decltype(std::declval<T>().Parameters())>;
  
  template<class>
  static auto has_parameters(long) -> std::false_type;
  
  template<typename T, typename U = void>
  struct reset_call
  {
    template<typename LayerType>
    static void call() {} 
  };
  
  template<typename T, typename U = void>
  struct reset_call_layer
  {
    template<typename LayerType>
    static void call() {} 
  };
  
  std::tuple<LayerTypes..., OutputLayerType> layers;
  
  template<typename LayerType>
  struct reset_call<LayerType, typename std::enable_if<decltype(has_parameters<LayerType>(0))::value>::type>
  {
    static void call(InitializationRuleType& initRule, LayerType& layer)
    {
      layer.Reset(initRule);
    }
  };
  
  template<typename LayerType>
  struct reset_call<LayerType, typename std::enable_if<!decltype(has_parameters<LayerType>(0))::value>::type>
  {
    static void call(InitializationRuleType& initRule, LayerType& layer)
    {
    
    }
  };
  
  template<typename T, size_t LayerNum>
  struct reset_call_layer<T, std::integral_constant<size_t, LayerNum>>
  {
    static void call(InitializationRuleType& initRule, std::tuple<LayerTypes..., OutputLayerType>& layers, NetworkType* network);
  };
  
  template<typename T>
  struct reset_call_layer<T, std::integral_constant<size_t, NumLayers>>
  {
    static void call(InitializationRuleType& initRule, std::tuple<LayerTypes..., OutputLayerType>& layers, NetworkType* network);
  };
  
  //! THe current target of the forward/backward pass.
  arma::mat currentTarget;
  
  arma::mat currentInput;
  
  //! The number of separable functions (the number of predictor points).
  size_t numFunctions;
  
  //! The matrix of data points (predictors).
  arma::mat predictors;

  //! The matrix of responses to the input data points.
  arma::mat responses;
  
  arma::mat parameter;
  
  arma::mat error;
};

template<typename InitializationRuleType, typename OutputLayerType, typename... LayerTypes>
auto BuildNetwork(const LayerTypes& ... layers) -> ANN<InitializationRuleType, sizeof...(LayerTypes), OutputLayerType, LayerTypes...>
{
  return ANN<InitializationRuleType, sizeof...(LayerTypes), OutputLayerType, LayerTypes...>(layers...);
}

}

}

// Include implementation.
#include "ann_impl.hpp"

#endif
