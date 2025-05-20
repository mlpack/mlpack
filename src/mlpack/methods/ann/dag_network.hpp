#ifndef MLPACK_METHODS_ANN_DAG_NETWORK_HPP
#define MLPACK_METHODS_ANN_DAG_NETWORK_HPP

#include "init_rules/init_rules.hpp"

namespace mlpack {

template<
    typename OutputLayerType = NegativeLogLikelihood,
    typename InitializationRuleType = RandomInitialization,
    typename MatType = arma::mat>
class DAGNetwork
{
public:
  DAGNetwork(OutputLayerType outputLayer = OutputLayerType(),
             InitializationRuleType initializeRule = InitializationRuleType());

  DAGNetwork(const DAGNetwork& other);
  DAGNetwork(DAGNetwork&& other);
  DAGNetwork& operator=(const DAGNetwork& other);
  DAGNetwork& operator=(DAGNetwork&& other);

  std::vector<size_t>& InputDimensions()
  {
    inputDimensionsAreSet = false;
    return inputDimensions;
  }

  const std::vector<size_t>& InputDimensions() const { return inputDimensions; }

  const std::vector<Layer<MatType>*>& Network() const
  {
    return network;
  }

  void Add(Layer<MatType>* layer) {
    network.push_back(layer);
  }

private:

  OutputLayerType outputLayer;
  InitializationRuleType initializeRule;

  bool inputDimensionsAreSet;
  std::vector<size_t> inputDimensions;

  std::vector<Layer<MatType>*> network;

};

} // namespace mlpack

#include "dag_network_impl.hpp"

#endif
