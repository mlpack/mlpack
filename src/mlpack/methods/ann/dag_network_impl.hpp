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
    network(network.network),
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
    network(std::move(network.network)),
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
    network = other.network;
    parameters = other.parameters;
    inputDimensions = other.inputDimensions;
    inputDimensionsAreSet = other.inputDimensionsAreSet;
    graphIsSet = other.graphIsSet;
    adjacencyList = other.adjacencyList;
    layerAxes = other.layerAxes;
    indices = other.indices;
    networkOutput = other.networkOutput;

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
    network = std::move(other.network);
    parameters = std::move(other.parameters);
    inputDimensions = std::move(other.inputDimensions);
    adjacencyList = std::move(other.adjacencyList);
    layerAxes = std::move(other.layerAxes);
    indices = std::move(other.indices);
    networkOutput = std::move(other.networkOutput);

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
  for (size_t i = 0; i < network.size(); i++)
    Log::Assert(layer != network[i], "DAGNetwork::Add(): Layer already exists.");

  network.push_back(layer);
  if (network.size() > 1)
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
  for (size_t i = 0; i < network.size(); i++)
    Log::Assert(layer != network[i], "DAGNetwork::Add(): Layer already exists.");

  network.push_back(layer);
  if (network.size() > 1)
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
  Log::Assert(inputLayer != outputLayer, "DAGNetwork::Connect(): `inputLayer` and `outputLayer` cannot be the same.");

  bool inputExists = false;
  bool outputExists = false;
  for (size_t i = 0; i < network.size(); i++)
  {
    if (network[i] == inputLayer)
    {
      inputExists = true;
    }
    else if (network[i] == outputLayer)
    {
      outputExists = true;
    }
  }

  Log::Assert(inputExists && outputExists, "DAGNetwork::Connect(): `inputLayer` and `outputLayer` must exist.");

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

  for (size_t i = 0; i < network.size(); i++)
  {
    size_t parents = 0;
    if (exploredLayers.count(network[i]))
      continue;
    exploring.push({network[i], false});

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
          Log::Assert(parent != network[i], "DAGNetwork::CheckGraph(): A cycle exists in the graph.");
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

  Log::Assert(size == sortedLayers.size(), "DAGNetwork::CheckGraph(): Multiple inputs and/or outputs exist when there should only be one of each");
  network = sortedLayers;
  graphIsSet = true;
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
  exploring.push(std::make_pair(network.back(), false));

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
      // TODO: this problem should be found in CheckGraph
      inputs++;
      Log::Assert(inputs == 1, "DAGNetwork::ComputeOutputDimensions(): There should only be one input node.");
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
        Log::Assert(layerAxes.count(currentLayer), "DAGNetwork::ComputeOutputDimensions(): Axis does not exist for an input into a concatenation.");

        size_t axis = layerAxes[currentLayer];
        const size_t numOutputDimensions = adjacencyList[currentLayer][0]->OutputDimensions().size();
        Log::Assert(axis > 0 && axis < numOutputDimensions,
        "DAGNetwork::ComputeOutputDimensions(): Axis to concatenate along must be within the "
                    "number of output dimensions.");
        for (size_t i = 1; i < numParents; i++)
        {
          Layer<MatType>* parent = adjacencyList[currentLayer][i];
          Log::Assert(numOutputDimensions == parent->OutputDimensions().size(),
          "DAGNetwork::ComputeOutputDimensions(): Number of output dimensions are not the same for "
                      "each input in the concatenation.");
        }
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
              Log::Assert(axisDim < axisDim2,
              "DAGNetwork::ComputeOutputDimensions(): Size of inputs dimensions not along the axis "
                          "dimension need to be the same.");
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
size_t DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::WeightSize()
{
  UpdateDimensions("DAGNetwork::WeightSize()");

  size_t total = 0;
  for (size_t i = 0; i < network.size(); i++)
    total += network[i]->WeightSize();
  return total;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Reset(const size_t inputDimensionality)
{
  parameters.clear();

  if (inputDimensionality != 0)
  {
    CheckNetwork("DAGNetwork::Reset()", inputDimensionality, true, false);
  }
  else if (inputDimensions.size() > 0)
  {
    size_t inputDim = inputDimensions[0];
    for (size_t i = 1; i < inputDimensions.size(); i++)
      inputDim *= inputDimensions[i];
    CheckNetwork("DAGNetwork::Reset()", inputDim, true, false);
  }
  else
  {
    throw std::invalid_argument("DAGNetwork::Reset(): cannot reset network when no "
        "input dimensionality is given, and `InputDimensions()` has not been "
        "set!");
  }
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
  for (size_t i = 0; i < network.size(); i++)
  {
    const size_t weightSize = network[i]->WeightSize();
    Log::Assert(offset + weightSize <= totalWeightSize,
        "DAGNetwork::SetLayerMemory(): Parameter size does not match total layer "
        "weight size!");
    MatType tmpWeights;
    MakeAlias(tmpWeights, weightsIn, weightSize, 1, offset);
    network[i]->SetWeights(tmpWeights);
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
  for (size_t i = 0; i < network.size(); ++i)
  {
    const size_t weightSize = network[i]->WeightSize();

    Log::Assert(start + weightSize <= totalWeightSize,
        "DAGNetwork::CustomInitialize(): Parameter size does not match total layer "
        "weight size!");

    MatType WTemp;
    MakeAlias(WTemp, W, weightSize, 1, start);
    network[i]->CustomInitialize(WTemp, weightSize);

    start += weightSize;
  }
  Log::Assert(start == totalWeightSize,
      "DAGNetwork::CustomInitialize(): Total layer weight size does not match rows "
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
  // Reset the network parameters with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(network, parameters);
  // Override the weight matrix if necessary.
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
      "DAGNetwork::SetLayerMemory(): Total layer weight size does not match parameter "
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
  for (size_t i = 0; i < network.size(); i++)
    network[i]->Training() = training;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::CheckNetwork(const std::string& functionName,
                const size_t inputDimensionality,
                const bool setMode,
                const bool training)
{
  if (network.size() == 0)
  {
    throw std::invalid_argument(functionName + ": cannot use network with no "
        "layers!");
  }

  if (!graphIsSet)
    CheckGraph();

  if (!inputDimensionsAreSet)
    UpdateDimensions(functionName, inputDimensionality);

  if (parameters.is_empty())
  {
    InitializeWeights();
  }
  else if (parameters.n_elem != WeightSize())
  {
    parameters.clear();
    InitializeWeights();
  }

  if (!layerMemoryIsSet)
    SetLayerMemory();

  if (setMode)
    SetNetworkMode(training);
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
  for (size_t i = 0; i < network.size() - 1; i++)
  {
    totalOutputSize += network[i]->OutputSize();
  }

  size_t totalConcatSize = 0;
  for (size_t i = 1; i < network.size(); i++)
  {
    if (adjacencyList[network[i]].size() > 1)
    {
      size_t concatSize = network[i]->InputDimensions()[0];
      for (size_t j = 1; j < network[i]->InputDimensions().size(); j++)
      {
        concatSize *= network[i]->InputDimensions()[j];
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
  for (size_t i = 0; i < network.size() - 1; i++)
  {
    const size_t layerOutputSize = network[i]->OutputSize();
    MakeAlias(layerOutputs[i], layerOutputMatrix, layerOutputSize, batchSize, offset);
    offset += batchSize * layerOutputSize;
    indices.insert({network[i], i});
  }

  //setup layerInputs
  for (size_t i = 1; i < network.size(); i++)
  {
    Layer<MatType>* currentLayer = network[i];
    std::vector<Layer<MatType>*> parents = adjacencyList[currentLayer];
    size_t numParents = parents.size();
    Log::Assert(numParents > 0,
        "DAGNetwork::InitializeForwardPassMemory(): Non-input nodes must have atleast one parent node.");
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
>::Forward(const MatType& input, MatType& results)
{
  CheckNetwork("DAGNetwork::Forward", input.n_rows);

  networkOutput.set_size(network.back()->OutputSize(), input.n_cols);

  // equivalent to MultiLayer->Forward()
  if (network.size() > 1)
  {
    InitializeForwardPassMemory(input.n_cols);
    network.front()->Forward(input, layerOutputs.front());
    for (size_t i = 1; i < network.size(); i++)
    {
      std::vector<Layer<MatType>*> parents = adjacencyList[network[i]];
      Log::Assert(parents.size() > 0,
        "DAGNetwork::Forward(): Non-input nodes must have atleast one parent node.");
      if (parents.size() > 1)
      {
        std::vector<CubeType> parentOutputAliases(parents.size());
        size_t axis = layerAxes[network[i]];

        size_t rows = 1;
        for (size_t j = 0; j < axis; j++)
          rows *= network[i]->InputDimensions()[j];

        size_t slices = input.n_cols;
        for (size_t j = axis + 1; j < network[i]->InputDimensions().size(); j++)
          slices *= network[i]->InputDimensions()[j];


        for (size_t j = 0; j < parents.size(); j++)
        {
          size_t cols = parents[j]->OutputDimensions()[axis];
          MatType& parentOutput = layerOutputs[indices[parents[j]]];
          MakeAlias(parentOutputAliases[j], parentOutput, rows, cols, slices);
        }

        CubeType inputAlias;
        MakeAlias(inputAlias, layerInputs[i-1], rows, network[i]->InputDimensions()[axis], slices);

        size_t startCol = 0;
        for (size_t j = 0; j < parentOutputAliases.size(); j++)
        {
          const size_t cols = parentOutputAliases[j].n_cols;
          inputAlias.cols(startCol, startCol + cols - 1) = parentOutputAliases[j];
          startCol += cols;
        }
      }
      if (i < network.size() - 1)
        network[i]->Forward(layerInputs[i-1], layerOutputs[i]);
      else
        network[i]->Forward(layerInputs[i-1], networkOutput);
    }
  }
  else if (network.size() == 1)
  {
    network[0]->Forward(input, networkOutput);
  }
  else
  {
    networkOutput = input;
  }

  if (&results != &networkOutput)
    results = networkOutput;
}


template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
double DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Loss() const
{
  double loss = 0.0;
  for (size_t i = 0; i < network.size(); i++)
       loss += network[i]->Loss();
  return loss;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Evaluate(const MatType& predictors, const MatType& responses)
{
  CheckNetwork("DAGNetwork::Evaluate()", predictors.n_rows);
  Forward(predictors, networkOutput);
  return outputLayer.Forward(networkOutput, responses) + Loss();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Predict(const MatType& predictors, MatType& results, const size_t batchSize)
{
  CheckNetwork("DAGNetwork::Predict()", predictors.n_rows, true, false);
  results.set_size(network.back()->OutputSize(), predictors.n_cols);
  for (size_t i = 0; i < predictors.n_cols; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols) - i);

    MatType predictorAlias, resultAlias;

    MakeAlias(predictorAlias, predictors, predictors.n_rows, effectiveBatchSize, i * predictors.n_rows);
    MakeAlias(resultAlias, results, results.n_rows, effectiveBatchSize, i * results.n_rows);

    Forward(predictorAlias, resultAlias);
  }
}

} // namespace mlpack

#endif
