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
  if (layers.size() > 1)
  {
    layerOutputs.push_back(MatType());
    layerInputs.push_back(MatType());
  }

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
  if (layers.size() > 1)
  {
    layerOutputs.push_back(MatType());
    layerInputs.push_back(MatType());
  }
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
      std::vector<Layer<MatType>*> parents = adjacencyList[currentLayer];

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
        for (size_t j = 0; j < parents.size(); j++)
        {
          Layer<MatType>* parent = parents[j];
          assert(parent != layers[i]  && "A cycle exists");
          if (!exploredLayers.count(parent))
          {
            exploring.push({parent, false});
          }
        }
      }
    }
  }

  size_t size = 0;
  exploring.push(std::make_pair(sortedLayers.back(), false));
  exploredLayers.clear();

  while (!exploring.empty())
  {
    auto [currentLayer, explored] = exploring.top();
    exploring.pop();
    std::vector<Layer<MatType>*> parents = adjacencyList[currentLayer];
    if (exploredLayers.count(currentLayer))
      continue;

    if (explored)
    {
      size++;
      exploredLayers.insert(currentLayer);
    }
    else
    {
      exploring.push({currentLayer, true});
      for (size_t j = 0; j < parents.size(); j++)
      {
        Layer<MatType>* parent = parents[j];
        if (!exploredLayers.count(parent))
        {
          exploring.push({parent, false});
        }
      }
    }

  }

  assert(size == sortedLayers.size() && "multiple sinks detected");
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
  std::stack<std::pair<Layer<MatType>*, bool>> exploring;
  std::unordered_set<Layer<MatType>*> explored;
  exploring.push(std::make_pair(layers.back(), false));

  size_t inputs = 0;
  while (!exploring.empty())
  {
    auto [currentLayer, alreadyExplored] = exploring.top();
    exploring.pop();
    size_t numParents = adjacencyList[currentLayer].size();

    if (explored.count(currentLayer))
    {
      assert(inputs++ == 0);
      continue;
    }

    if (numParents == 0)
    {
      currentLayer->InputDimensions() = inputDimensions;
      explored.insert(currentLayer);
    }
    else if (alreadyExplored)
    {
      if (numParents == 1)
      {
        currentLayer->InputDimensions() = adjacencyList[currentLayer][0]->OutputDimensions();
        explored.insert(currentLayer);
      }
      else
      {
        // compute concat dimensions
        explored.insert(currentLayer);
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

        for (size_t i = 0; i < currentLayer->InputDimensions().size(); i++)
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
      currentLayer->ComputeOutputDimensions();
    }
    else
    {
      exploring.push(std::make_pair(currentLayer, true));
      for (size_t i = 0; i < numParents; i++)
      {
        exploring.push(std::make_pair(adjacencyList[currentLayer][i], false));
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


template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
const size_t DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::WeightSize() const
{
  // FIXME: assumed toposorted

  size_t total = 0;
  for (size_t i = 0; i < layers.size(); i++)
    total += layers[i]->WeightSize();
  return total;

}
template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::SetWeights(const MatType& weightsIn)
{
  // FIXME: assumed toposorted

  size_t offset = 0;
  const size_t totalWeightSize = WeightSize();
  for (size_t i = 0; i < layers.size(); i++)
  {
    const size_t weightSize = layers[i]->WeightSize();
    assert(weightSize + offset <= totalWeightSize);
    MatType tmpWeights;
    MakeAlias(tmpWeights, weightsIn, weightSize, 1, offset);
    layers[i]->SetWeights(tmpWeights);
    offset += weightSize;
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::CustomInitialize(
    MatType& W,
    const size_t elements)
{
  size_t start = 0;
  const size_t totalWeightSize = elements;
  for (size_t i = 0; i < layers.size(); ++i)
  {
    const size_t weightSize = layers[i]->WeightSize();

    Log::Assert(start + weightSize <= totalWeightSize,
        "FNN::CustomInitialize(): parameter size does not match total layer "
        "weight size!");

    MatType WTemp;
    MakeAlias(WTemp, W, weightSize, 1, start);
    layers[i]->CustomInitialize(WTemp, weightSize);

    start += weightSize;
  }
  Log::Assert(start == totalWeightSize,
      "FNN::CustomInitialize(): total layer weight size does not match rows "
      "size!");
}
template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::InitializeWeights()
{
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(layers, parameters);
  CustomInitialize(parameters, WeightSize());
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::SetLayerMemory()
{
  size_t totalWeightSize = WeightSize();

  Log::Assert(totalWeightSize == parameters.n_elem,
      "FFN::SetLayerMemory(): total layer weight size does not match parameter "
      "size!");

  SetWeights(parameters);
  layerMemoryIsSet = true;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::SetNetworkMode(const bool training)
{
  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->Training() = training;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::InitializeForwardPassMemory(const size_t batchSize)
{
  // assumed toposorted and computeoutputdimensions
  size_t totalOutputSize = 0;
  for (size_t i = 0; i < layers.size() - 1; i++)
  {
    totalOutputSize += layers[i]->OutputSize();
  }

  size_t totalConcatSize = 0;
  for (size_t i = 1; i < layers.size(); i++)
  {
    if (adjacencyList[layers[i]].size() > 1)
    {
      size_t concatSize = layers[i]->InputDimensions()[0];
      for (size_t j = 1; j < layers[i]->InputDimensions().size(); j++)
      {
        concatSize += layers[i]->InputDimensions()[j];
      }
      totalConcatSize += concatSize;
    }
  }

  size_t forwardMemSize = totalOutputSize + totalConcatSize;
  if (batchSize * forwardMemSize >
    layerOutputMatrix.n_elem)
  {
    layerOutputMatrix = MatType(1, batchSize * forwardMemSize);
  }

  size_t offset = 0;
  indices.clear();
  //setup layerOutputs
  for (size_t i = 0; i < layers.size() - 1; i++)
  {
    const size_t layerOutputSize = layers[i]->OutputSize();
    MakeAlias(layerOutputs[i], layerOutputMatrix, layerOutputSize, batchSize, offset);
    offset += batchSize * layerOutputSize;
    indices.insert({layers[i], i});
  }

  //setup layerInputs
  for (size_t i = 1; i < layers.size(); i++)
  {
    Layer<MatType>* currentLayer = layers[i];
    std::vector<Layer<MatType>*> parents = adjacencyList[currentLayer];
    size_t numParents = parents.size();
    assert(numParents > 0);
    if (numParents == 1)
    {
      size_t inputIndex = indices[parents.front()];
      MakeAlias(layerInputs[i-1], layerOutputs[inputIndex], layerOutputs[inputIndex].n_rows, layerOutputs[inputIndex].n_cols, 0);
    }
    else
    {
      size_t concatSize = currentLayer->InputDimensions()[0];
      for (size_t j = 1; j < currentLayer->InputDimensions().size(); j++)
        concatSize += currentLayer->InputDimensions()[j];
      MakeAlias(layerInputs[i-1], layerOutputMatrix, concatSize, batchSize, offset);
      offset += concatSize;
    }
  }
}

} // namespace mlpack

#endif
