#ifndef MLPACK_METHODS_ANN_DAG_NETWORK_IMPL_HPP
#define MLPACK_METHODS_ANN_DAG_NETWORK_IMPL_HPP

#include "dag_network.hpp"

namespace mlpack {

template<typename OutputLayerType,
         typename InitializationRuleType, 
         typename MatType>
DAGNetwork<
    OutputLayerType,
    InitializationRuleType, 
    MatType
>::DAGNetwork(OutputLayerType outputLayer,
              InitializationRuleType initializeRule) :
    outputLayer(outputLayer),
    initializeRule(initializeRule),
    inputDimensionsAreSet(false)
{}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::DAGNetwork(const DAGNetwork& network) :
    outputLayer(network.outputLayer),
    initializeRule(network.initializeRule),
    inputDimensionsAreSet(false)
{}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::DAGNetwork(DAGNetwork&& network) :
    outputLayer(std::move(network.outputLayer)),
    initializeRule(std::move(network.initializeRule)),
    inputDimensionsAreSet(std::move(network.inputDimensionsAreSet))
{}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
DAGNetwork<OutputLayerType,
           InitializationRuleType,
           MatType>&
DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::operator=(const DAGNetwork& other)
{
  if (this != &other)
  {
    outputLayer = other.outputLayer;
    initializeRule = other.initializeRule;
    inputDimensionsAreSet = other.inputDimensionsAreSet;
  }

  return *this;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
DAGNetwork<OutputLayerType,
           InitializationRuleType,
           MatType>& DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::operator=(DAGNetwork&& other)
{
  if (this != &other)
  {
    outputLayer = std::move(other.outputLayer);
    initializeRule = std::move(other.initializeRule);
    inputDimensionsAreSet = std::move(other.inputDimensionsAreSet);
  }

  return *this;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Add(Layer<MatType>* layer)
{
  // check if layer exists in layers?
  layers.push_back(layer);
  inputDimensionsAreSet = false;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Add(Layer<MatType>* layer, size_t axis)
{
  layers.push_back(layer);
  layerAxes[layer] = axis;
  inputDimensionsAreSet = false;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Connect(Layer<MatType>* inputLayer, Layer<MatType>* outputLayer)
{
  if (adjacencyList.count(outputLayer) == 0)
  {
    adjacencyList.insert({outputLayer, {inputLayer}});
  }
  else
  {
    adjacencyList[outputLayer].push_back(inputLayer);
  }
  inputDimensionsAreSet = false;
}

// topo sort, no cycles,
// network has one output
template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::CheckGraph()
{
  std::unordered_set<Layer<MatType>*> exploredLayers;
  std::vector<Layer<MatType>*> sortedLayers;
  std::stack<std::pair<Layer<MatType>*, bool>> exploring;

  for (size_t i = 0; i < layers.size(); i++)
  {
    size_t parents = 0;
    if (exploredLayers.count(layers[i]))
      continue;
    exploring.push({layers[i], false});

    while (!exploring.empty())
    {
      auto [currentLayer, explored] = exploring.top();
      exploring.pop();

      if (exploredLayers.count(currentLayer))
          continue;

      if (explored)
      {
        sortedLayers.push_back(currentLayer);
        exploredLayers.insert(currentLayer);
      }
      else
      {
        exploring.push({currentLayer, true});
        for (size_t j = 0; j < adjacencyList[currentLayer].size(); j++)
        {
          Layer<MatType>* parent = adjacencyList[currentLayer][j];
          assert(parent != layers[i]  && "A cycle exists");
          if (!exploredLayers.count(parent))
          {
            exploring.push({parent, false});
          }
        }
      }
    }
  }

  size_t totalParents = 0;
  std::stack<Layer<MatType>*> parents;
  parents.push(sortedLayers.back());

  bool parentsExist = true;
  while (parentsExist)
  {
    Layer<MatType>* currentLayer = parents.top();
    parents.pop();

    size_t numParents = adjacencyList[currentLayer].size();
    if (numParents == 0)
    {
      parentsExist = false;
    }
    else
    {
      totalParents += numParents;
      for (size_t i = 0; i < numParents; i++)
      {
          parents.push(adjacencyList[currentLayer][i]);
      }
    }
  }

  assert(totalParents == sortedLayers.size() - 1 && "multiple sinks detected.");
  layers = sortedLayers;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::ComputeOutputDimensions()
{
  // assuming topological sorted layers
  std::stack<std::pair<Layer<MatType>*, bool>> exploredParents;
  exploredParents.push(std::make_pair(layers.back(), false));

  while (!exploredParents.empty())
  {
    auto [currentLayer, explored] = exploredParents.top();
    exploredParents.pop();
    size_t numParents = adjacencyList[currentLayer].size();

    if (numParents == 0)
    {
      currentLayer->InputDimensions() = inputDimensions;
    }
    else if (explored)
    {
      if (numParents == 1)
      {
        currentLayer->InputDimensions() = adjacencyList[currentLayer][0]->OutputDimensions();
      }
      else
      {
        // compute concat dimensions
        assert(layerAxes.count(currentLayer) && "Axis does not exist for a skip connection");

        size_t axis = layerAxes[currentLayer];
        const size_t numOutputDimensions = adjacencyList[currentLayer][0]->OutputDimensions().size();
        for (size_t i = 1; i < numParents; i++)
        {
          Layer<MatType>* parent = adjacencyList[currentLayer][i];
          assert(numOutputDimensions == parent->OutputDimensions().size());
        }
        assert(axis < numOutputDimensions);
        currentLayer->InputDimensions() = std::vector<size_t>(numOutputDimensions, 0);

        for (size_t i = 0; i < currentLayer->OutputDimensions().size(); i++)
        {
          if (i == axis)
          {
            for (size_t n = 0; n < numParents; n++)
            {
              Layer<MatType>* parent = adjacencyList[currentLayer][n];
              currentLayer->InputDimensions()[i] += parent->OutputDimensions()[i];
            }
          }
          else
          {
            Layer<MatType>* firstParent = adjacencyList[currentLayer].front();
            const size_t axisDim = firstParent->OutputDimensions()[i];
            for (size_t n = 1; n < adjacencyList[currentLayer].size(); n++)
            {
              Layer<MatType>* parent = adjacencyList[currentLayer][n];
              const size_t axisDim2 = parent->OutputDimensions()[i];
              assert(axisDim == axisDim2);
            }
            currentLayer->InputDimensions()[i] = axisDim;
          }
        }
      }
    }
    else
    {
      exploredParents.push(std::make_pair(currentLayer, true));
      for (size_t i = 0; i < numParents; i++)
      {
        exploredParents.push(std::make_pair(adjacencyList[currentLayer][i], false));
      }
    }
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::UpdateDimensions(const std::string& functionName,
                    const size_t inputDimensionality)
{
  if (inputDimensions.size() == 0)
    inputDimensions = { inputDimensionality };

  size_t totalInputSize = 1;
  for (size_t i = 0; i < inputDimensions.size(); i++)
    totalInputSize *= inputDimensions[i];

  if (totalInputSize != inputDimensionality && inputDimensionality != 0)
  {
    throw std::logic_error(functionName + ": input size does not match expected size set with InputDimensions()!");
  }

  ComputeOutputDimensions();
  inputDimensionsAreSet = true;
}

} // namespace mlpack

#endif
