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
>::Connect(Layer<MatType>* inputLayer, Layer<MatType>* outputLayer)
{
    if (adjacencyList.count(outputLayer) == 0) {
        adjacencyList.insert({outputLayer, {inputLayer}});
    } else {
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

    while (parentsExist) {
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

} // namespace mlpack

#endif
