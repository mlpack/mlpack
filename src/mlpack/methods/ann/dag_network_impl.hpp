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
    // These will be set correctly in the first Forward() call.
    inputDimensionsAreSet(false),
    layerMemoryIsSet(false),
    graphIsSet(false)
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
    layers(network.layers),
    parameters(network.parameters),
    inputDimensions(network.inputDimensions),
    adjacencyList(network.adjacencyList),
    layerAxes(network.layerAxes),
    indices(network.indices),
    // These will be set correctly in the first Forward() call.
    inputDimensionsAreSet(false),
    layerMemoryIsSet(false),
    graphIsSet(false)
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
    layers(std::move(network.layers)),
    parameters(std::move(network.parameters)),
    inputDimensions(std::move(network.inputDimensions)),
    adjacencyList(std::move(network.adjacencyList)),
    layerAxes(std::move(network.layerAxes)),
    indices(std::move(network.indices)),
    // Aliases will not be correct after a std::move(), so we will manually
    // reset them.
    layerMemoryIsSet(false),
    graphIsSet(std::move(network.graphIsSet)),
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
    layers = other.layers;
    parameters = other.parameters;
    inputDimensions = other.inputDimensions;
    inputDimensionsAreSet = other.inputDimensionsAreSet;
    graphIsSet = other.graphIsSet;
    adjacencyList = other.adjacencyList;
    layerAxes = other.layerAxes;
    indices = other.indices;

    layerMemoryIsSet = false;
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
    layers = std::move(other.layers);
    parameters = std::move(other.parameters);
    inputDimensions = std::move(other.inputDimensions);
    adjacencyList = std::move(other.adjacencyList);
    layerAxes = std::move(other.layerAxes);
    indices = std::move(other.indices);

    inputDimensionsAreSet = std::move(other.inputDimensionsAreSet);
    graphIsSet = std::move(other.graphIsSet);
    layerMemoryIsSet = false;
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
  // check if layer exists in layers
  for (size_t i = 0; i < layers.size(); i++)
    assert(layer != layers[i]);

  layers.push_back(layer);
  if (layers.size() > 1)
  {
    layerOutputs.push_back(MatType());
    layerInputs.push_back(MatType());
  }

  inputDimensionsAreSet = false;
  graphIsSet = false;
  layerMemoryIsSet = false;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Add(Layer<MatType>* layer, size_t concatAxis)
{
  // check if layer exists in layers
  for (size_t i = 0; i < layers.size(); i++)
    assert(layer != layers[i]);

  layers.push_back(layer);
  if (layers.size() > 1)
  {
    layerOutputs.push_back(MatType());
    layerInputs.push_back(MatType());
  }
  layerAxes[layer] = concatAxis;

  inputDimensionsAreSet = false;
  graphIsSet = false;
  layerMemoryIsSet = false;
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
  assert(inputLayer != outputLayer);

  // check if inputLayer and outputlayer exist in layers
  bool inputExists = false;
  bool outputExists = false;
  for (size_t i = 0; i < layers.size(); i++)
  {
    if (layers[i] == inputLayer)
    {
      inputExists = true;
    }
    else if (layers[i] == outputLayer)
    {
      outputExists = true;
    }
  }
  assert(inputExists && outputExists);

  if (adjacencyList.count(outputLayer) == 0)
  {
    adjacencyList.insert({outputLayer, {inputLayer}});
  }
  else
  {
    adjacencyList[outputLayer].push_back(inputLayer);
  }

  inputDimensionsAreSet = false;
  graphIsSet = false;
  layerMemoryIsSet = false;
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
  if (!graphIsSet)
    CheckGraph();
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
      continue;
    }

    if (numParents == 0)
    {
      assert(inputs++ == 0);
      currentLayer->InputDimensions() = inputDimensions;
      currentLayer->ComputeOutputDimensions();
      explored.insert(currentLayer);
    }
    else if (alreadyExplored)
    {
      explored.insert(currentLayer);
      if (numParents == 1)
      {
        currentLayer->InputDimensions() = adjacencyList[currentLayer][0]->OutputDimensions();
      }
      else
      {
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
>::WeightSize()
{
  UpdateDimensions("DAGNetwork::WeightSize()");

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
  if (!graphIsSet)
    CheckGraph();

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
  if (!graphIsSet)
    CheckGraph();

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
        concatSize *= layers[i]->InputDimensions()[j];
      }
      totalConcatSize += concatSize;
    }
  }

  size_t forwardMemSize = totalOutputSize + totalConcatSize;
  if (batchSize * forwardMemSize > layerOutputMatrix.n_elem)
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
        concatSize *= currentLayer->InputDimensions()[j];
      MakeAlias(layerInputs[i-1], layerOutputMatrix, concatSize, batchSize, offset);
      offset += concatSize;
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
>::Forward(const MatType& input, MatType& output)
{
  output.set_size(layers.back()->OutputSize(), input.n_cols);

  if (layers.size() > 1)
  {
    InitializeForwardPassMemory(input.n_cols);
    layers.front()->Forward(input, layerOutputs.front());
    for (size_t i = 1; i < layers.size(); i++)
    {
      std::vector<Layer<MatType>*> parents = adjacencyList[layers[i]];
      assert(parents.size() > 0);
      if (parents.size() > 1)
      {
        parentOutputAliases.clear();
        parentOutputAliases.resize(parents.size());
        size_t axis = layerAxes[layers[i]];

        size_t rows = 1;
        for (size_t j = 0; j < axis; j++)
          rows *= layers[i]->InputDimensions()[j];

        size_t slices = input.n_cols;
        for (size_t j = axis + 1; j < layers[i]->InputDimensions().size(); j++)
          slices *= layers[i]->InputDimensions()[j];


        for (size_t j = 0; j < parents.size(); j++)
        {
          size_t cols = parents[j]->OutputDimensions()[axis];
          MatType& parentOutput = layerOutputs[indices[parents[j]]];
          MakeAlias(parentOutputAliases[j], parentOutput, rows, cols, slices);
        }

        MakeAlias(inputAlias, layerInputs[i-1], rows, layers[i]->InputDimensions()[axis], slices);

        size_t startCol = 0;
        for (size_t j = 0; j < parentOutputAliases.size(); j++)
        {
          const size_t cols = parentOutputAliases[j].n_cols;
          inputAlias.cols(startCol, startCol + cols - 1) = parentOutputAliases[j];
          startCol += cols;
        }
      }
      if (i < layers.size() - 1)
        layers[i]->Forward(layerInputs[i-1], layerOutputs[i]);
      else
        layers[i]->Forward(layerInputs[i-1], output);
    }
  }
  else if (layers.size() == 1)
  {
    layers[0]->Forward(input, output);
  }
  else
  {
    output = input;
  }
}

} // namespace mlpack

#endif
