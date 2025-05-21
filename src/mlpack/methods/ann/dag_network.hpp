#ifndef MLPACK_METHODS_ANN_DAG_NETWORK_HPP
#define MLPACK_METHODS_ANN_DAG_NETWORK_HPP

#include <mlpack/core.hpp>

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

  const std::vector<Layer<MatType>*>& Network() {
    CheckGraph();
    return layers;
  }

  void Add(Layer<MatType>* layer);

  void Add(Layer<MatType>* layer, size_t axis);

  void Connect(Layer<MatType>* inputLayer, Layer<MatType>* outputLayer);

  // topo sort, no cycles, network has one output
  void CheckGraph();

// private:
  OutputLayerType outputLayer;
  InitializationRuleType initializeRule;

  bool inputDimensionsAreSet;
  std::vector<size_t> inputDimensions;

  std::vector<Layer<MatType>*> layers;
  std::map<Layer<MatType>*, std::vector<Layer<MatType>*>> adjacencyList;
  std::map<Layer<MatType>*, size_t> layerAxes;
};

} // namespace mlpack

#include "dag_network_impl.hpp"

#endif
