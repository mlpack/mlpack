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
    graphIsSet = false;
    return inputDimensions;
  }

  const std::vector<size_t>& InputDimensions() const { return inputDimensions; }

  const std::vector<Layer<MatType>*>& Network()
  {
    if (!graphIsSet)
      CheckGraph();
    return layers;
  }

  void Add(Layer<MatType>* layer);

  void Add(Layer<MatType>* layer, size_t axis);

  void Connect(Layer<MatType>* inputLayer, Layer<MatType>* outputLayer);

  void ComputeOutputDimensions();

  void UpdateDimensions(const std::string& functionName,
                        const size_t inputDimensionality = 0);

  // topo sort, no cycles, network has one output
  void CheckGraph();

  const size_t WeightSize() const;
  void SetWeights(const MatType& weightsIn);

  void InitializeWeights();
  void CustomInitialize(MatType& W, const size_t elements);

  void SetLayerMemory();

  void SetNetworkMode(const bool training);

  using CubeType = typename GetCubeType<MatType>::type;
  void InitializeForwardPassMemory(const size_t batchSize);
  void Forward(const MatType& input, MatType& output);

// private:
  OutputLayerType outputLayer;
  InitializationRuleType initializeRule;

  bool inputDimensionsAreSet;
  bool graphIsSet;
  bool layerMemoryIsSet;

  std::vector<size_t> inputDimensions;

  std::vector<Layer<MatType>*> layers;
  std::map<Layer<MatType>*, std::vector<Layer<MatType>*>> adjacencyList;
  std::map<Layer<MatType>*, size_t> layerAxes;
  std::map<Layer<MatType>*, size_t> indices; // layer, i (i == where in toposorted layers)

  MatType parameters;

  CubeType inputAlias;
  std::vector<CubeType> parentOutputAliases;

  MatType layerOutputMatrix;
  std::vector<MatType> layerInputs;
  std::vector<MatType> layerOutputs;
};

} // namespace mlpack

#include "dag_network_impl.hpp"

#endif
