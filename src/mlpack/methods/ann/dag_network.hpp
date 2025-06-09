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
    layerMemoryIsSet = false;

    return inputDimensions;
  }

  const std::vector<size_t>& InputDimensions() const { return inputDimensions; }

  const MatType& Parameters() const { return parameters; }
  MatType& Parameters() { return parameters; }

  void Reset(const size_t inputDimensionality = 0);

  const std::vector<Layer<MatType>*>& Network()
  {
    if (!graphIsSet)
      CheckGraph();
    return network;
  }

  void Add(Layer<MatType>* layer);

  template <typename LayerType, typename... Args>
  void Add(Args... args);

  void Add(Layer<MatType>* layer, size_t concatAxis);

  void Connect(Layer<MatType>* inputLayer, Layer<MatType>* outputLayer);

  void ComputeOutputDimensions();

  void UpdateDimensions(const std::string& functionName,
                        const size_t inputDimensionality = 0);

  // topo sorts network, no cycles, network has one input and one output
  void CheckGraph();

  void CheckNetwork(const std::string& functionName,
                    const size_t inputDimensionality,
                    const bool setMode = false,
                    const bool training = false);

  const size_t WeightSize();
  void SetWeights(const MatType& weightsIn);

  void InitializeWeights();
  void CustomInitialize(MatType& W, const size_t elements);

  void SetLayerMemory();

  void SetNetworkMode(const bool training);

  using CubeType = typename GetCubeType<MatType>::type;
  void InitializeForwardPassMemory(const size_t batchSize);
  double Loss() const;

  void Forward(const MatType& input, MatType& output);

// private:
  OutputLayerType outputLayer;
  InitializationRuleType initializeRule;

  MatType networkOutput;

  bool inputDimensionsAreSet;
  bool graphIsSet;
  bool layerMemoryIsSet;

  std::vector<size_t> inputDimensions;

  std::vector<Layer<MatType>*> network;
  std::map<Layer<MatType>*, std::vector<Layer<MatType>*>> adjacencyList;
  std::map<Layer<MatType>*, size_t> layerAxes;
  std::map<Layer<MatType>*, size_t> indices; // layer, i (i == where in toposorted network)

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
