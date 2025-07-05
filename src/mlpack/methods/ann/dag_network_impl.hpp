/**
 * @file methods/ann/dag_network_impl.hpp
 * @author Andrew Furey
 *
 * Definition of the DAGNetwork class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
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
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    // These will be set correctly in the first Forward() call.
    validOutputDimensions(false),
    layerMemoryIsSet(false),
    graphIsSet(false)
{
  /* Nothing to do here. */
}

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
    sortedNetwork(network.sortedNetwork),
    sortedIndices(network.sortedIndices),
    // These will be set correctly in the first Forward() call.
    validOutputDimensions(false),
    layerMemoryIsSet(false),
    graphIsSet(false)
{
  /* Nothing to do here. */
}

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
    sortedNetwork(std::move(network.sortedNetwork)),
    sortedIndices(std::move(network.sortedIndices)),
    // Aliases will not be correct after a std::move(), so we will manually
    // reset them.
    layerMemoryIsSet(false),
    graphIsSet(std::move(network.graphIsSet)),
    validOutputDimensions(std::move(network.validOutputDimensions))
{
  /* Nothing to do here. */
}

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
    adjacencyList = other.adjacencyList;
    layerAxes = other.layerAxes;
    sortedNetwork = other.sortedNetwork;
    sortedIndices = other.sortedIndices;
    networkOutput = other.networkOutput;

    validOutputDimensions = false;
    graphIsSet = false;
    layerMemoryIsSet = false;

    layerInputs.clear();
    layerOutputs.clear();
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
    sortedNetwork = std::move(other.sortedNetwork);
    sortedIndices = std::move(other.sortedIndices);
    networkOutput = std::move(other.networkOutput);

    validOutputDimensions = false;
    graphIsSet = false;
    layerMemoryIsSet = false;

    layerInputs.clear();
    layerOutputs.clear();
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
>::SetAxis(size_t layerId, size_t concatAxis)
{
  if (layerId >= network.size())
  {
    throw std::logic_error("DAGNetwork::SetAxis(): layer does not exist in the network.");
  }

  if (layerAxes.find(network[layerId]) != layerAxes.end())
  {
    throw std::logic_error("DAGNetwork::SetAxis(): Cannot change concatenation axis.");
  }
  layerAxes[network[layerId]] = concatAxis;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Connect(size_t parentNodeId, size_t childNodeId)
{
  if (parentNodeId == childNodeId)
  {
    throw std::logic_error("DAGNetwork::Connect(): `inputLayer` and `outputLayer` cannot be the same.");
  }

  if (parentNodeId >= network.size() || childNodeId >= network.size())
  {
    throw std::logic_error("DAGNetwork::Connect(): `inputLayer` and `outputLayer` must exist in the network before connecting them.");
  }

  if (adjacencyList.count(network[childNodeId]) == 0)
  {
    adjacencyList.insert({network[childNodeId], {network[parentNodeId]}});
  }
  else
  {
    const std::vector<Layer<MatType>*>& list = adjacencyList[network[childNodeId]];
    for (size_t i = 0; i < list.size(); i++)
    {
      if (list[i] == network[parentNodeId])
        throw std::logic_error("DAGNetwork::Connect(): Cannot concatenate the outputs of the same layer.");
    }
    list.push_back(network[parentNodeId]);
  }

  validOutputDimensions = false;
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
  std::stack<std::pair<Layer<MatType>*, bool>> exploring;
  sortedNetwork.clear();

  size_t inputLayers = 0;

  for (size_t i = 0; i < network.size(); i++)
  {
    size_t parents = 0;
    if (exploredLayers.count(network[i]))
      continue;
    exploring.push({network[i], false});

    if (adjacencyList[network[i]].size() == 0)
    {
      inputLayers++;
      if (inputLayers > 1)
      {
        throw std::logic_error("DAGNetwork::CheckGraph(): There should only be one input node.");
      }
    }

    while (!exploring.empty())
    {
      auto [currentLayer, explored] = exploring.top();
      exploring.pop();
      const std::vector<Layer<MatType>*>& parents = adjacencyList[currentLayer];

      if (exploredLayers.count(currentLayer))
          continue;

      if (explored)
      {
        sortedNetwork.push_back(currentLayer);
        exploredLayers.insert(currentLayer);
      }
      else
      {
        exploring.push({currentLayer, true});
        for (size_t j = 0; j < parents.size(); j++)
        {
          Layer<MatType>* parent = parents[j];
          if (parent == network[i])
          {
            throw std::logic_error("DAGNetwork::CheckGraph(): A cycle exists in the graph.");
          }

          if (!exploredLayers.count(parent))
          {
            exploring.push({parent, false});
          }
        }
      }
    }
  }

  size_t size = 0;
  exploring.push(std::make_pair(sortedNetwork.back(), false));
  exploredLayers.clear();

  while (!exploring.empty())
  {
    auto [currentLayer, explored] = exploring.top();
    exploring.pop();
    const std::vector<Layer<MatType>*>& parents = adjacencyList[currentLayer];
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

  if (size != sortedNetwork.size())
  {
    throw std::logic_error("DAGNetwork::CheckGraph(): Multiple inputs and/or outputs exist when there should only be one of each");
  }
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
  exploring.push(std::make_pair(sortedNetwork.back(), false));

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
        if (layerAxes.count(currentLayer) == 0)
        {
          throw std::logic_error("DAGNetwork::ComputeOutputDimensions(): Axis does not exist for an input into a concatenation.");
        }

        size_t axis = layerAxes[currentLayer];
        const size_t numOutputDimensions = adjacencyList[currentLayer][0]->OutputDimensions().size();

        if (axis < 0 || axis >= numOutputDimensions)
        {
          throw std::logic_error("DAGNetwork::ComputeOutputDimensions(): Axis to concatenate along must be within the number of output dimensions.");
        }

        for (size_t i = 1; i < numParents; i++)
        {
          Layer<MatType>* parent = adjacencyList[currentLayer][i];

          if (numOutputDimensions != parent->OutputDimensions().size())
          {
            throw std::logic_error("DAGNetwork::ComputeOutputDimensions(): Number of output dimensions are not the same for "
                      "each input in the concatenation.");
          }
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

              if (axisDim != axisDim2)
              {
                throw std::logic_error("DAGNetwork::ComputeOutputDimensions(): Size of inputs dimensions that are not along the concatenation axis "
                          "dimension need to be the same.");
              }
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
  validOutputDimensions = true;
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
  for (size_t i = 0; i < sortedNetwork.size(); i++)
    total += sortedNetwork[i]->WeightSize();
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
  for (size_t i = 0; i < sortedNetwork.size(); i++)
  {
    const size_t weightSize = sortedNetwork[i]->WeightSize();

    if (offset + weightSize > totalWeightSize)
    {
      throw std::logic_error("DAGNetwork::SetLayerMemory(): Parameter size does not match total layer "
        "weight size!");
    }
    MatType tmpWeights;
    MakeAlias(tmpWeights, weightsIn, weightSize, 1, offset);
    sortedNetwork[i]->SetWeights(tmpWeights);
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
  if (!graphIsSet)
    CheckGraph();

  size_t offset = 0;
  const size_t totalWeightSize = elements;
  for (size_t i = 0; i < sortedNetwork.size(); ++i)
  {
    const size_t weightSize = sortedNetwork[i]->WeightSize();
    if (offset + weightSize > totalWeightSize)
    {
      throw std::logic_error("DAGNetwork::SetLayerMemory(): Parameter size does not match total layer "
        "weight size!");
    }

    MatType WTemp;
    MakeAlias(WTemp, W, weightSize, 1, offset);
    sortedNetwork[i]->CustomInitialize(WTemp, weightSize);

    offset += weightSize;
  }

  if (offset != totalWeightSize)
  {
    throw std::logic_error("DAGNetwork::CustomInitialize(): Total layer weight size does not match rows "
      "size!");
  }
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
  networkInit.Initialize(sortedNetwork, parameters);
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

  if (totalWeightSize != parameters.n_elem)
  {
    throw std::logic_error("DAGNetwork::SetLayerMemory(): Total layer weight size does not match parameter "
      "size!");
  }

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

  if (!validOutputDimensions)
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
  CheckNetwork("DAGNetwork::InitializeForwardPassMemory()", 0);

  size_t totalOutputSize = 0;
  for (size_t i = 0; i < sortedNetwork.size() - 1; i++)
  {
    totalOutputSize += sortedNetwork[i]->OutputSize();
  }

  size_t totalConcatSize = 0;
  for (size_t i = 1; i < sortedNetwork.size(); i++)
  {
    if (adjacencyList[sortedNetwork[i]].size() > 1)
    {
      size_t concatSize = sortedNetwork[i]->InputDimensions()[0];
      for (size_t j = 1; j < sortedNetwork[i]->InputDimensions().size(); j++)
      {
        concatSize *= sortedNetwork[i]->InputDimensions()[j];
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
  sortedIndices.clear();
  //setup layerOutputs
  for (size_t i = 0; i < sortedNetwork.size() - 1; i++)
  {
    const size_t layerOutputSize = sortedNetwork[i]->OutputSize();
    MakeAlias(layerOutputs[i], layerOutputMatrix, layerOutputSize, batchSize, offset);
    offset += batchSize * layerOutputSize;
    sortedIndices.insert({sortedNetwork[i], i});
  }

  //setup layerInputs
  for (size_t i = 1; i < sortedNetwork.size(); i++)
  {
    Layer<MatType>* currentLayer = sortedNetwork[i];
    const std::vector<Layer<MatType>*>& parents = adjacencyList[currentLayer];
    size_t numParents = parents.size();

    if (numParents == 1)
    {
      size_t inputIndex = sortedIndices[parents.front()];
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
  CheckNetwork("DAGNetwork::Forward()", input.n_rows);

  networkOutput.set_size(sortedNetwork.back()->OutputSize(), input.n_cols);

  // equivalent to MultiLayer->Forward(), but with a concat.
  if (sortedNetwork.size() > 1)
  {
    InitializeForwardPassMemory(input.n_cols);
    sortedNetwork.front()->Forward(input, layerOutputs.front());
    for (size_t i = 1; i < sortedNetwork.size(); i++)
    {
      const std::vector<Layer<MatType>*>& parents = adjacencyList[sortedNetwork[i]];

      if (parents.size() > 1)
      {
        std::vector<CubeType> parentOutputAliases(parents.size());
        size_t axis = layerAxes[sortedNetwork[i]];

        size_t rows = 1;
        for (size_t j = 0; j < axis; j++)
          rows *= sortedNetwork[i]->InputDimensions()[j];

        size_t slices = input.n_cols;
        for (size_t j = axis + 1; j < sortedNetwork[i]->InputDimensions().size(); j++)
          slices *= sortedNetwork[i]->InputDimensions()[j];


        for (size_t j = 0; j < parents.size(); j++)
        {
          size_t cols = parents[j]->OutputDimensions()[axis];
          MatType& parentOutput = layerOutputs[sortedIndices[parents[j]]];
          MakeAlias(parentOutputAliases[j], parentOutput, rows, cols, slices);
        }

        CubeType inputAlias;
        MakeAlias(inputAlias, layerInputs[i-1], rows, sortedNetwork[i]->InputDimensions()[axis], slices);

        size_t startCol = 0;
        for (size_t j = 0; j < parentOutputAliases.size(); j++)
        {
          const size_t cols = parentOutputAliases[j].n_cols;
          inputAlias.cols(startCol, startCol + cols - 1) = parentOutputAliases[j];
          startCol += cols;
        }
      }

      if (i < sortedNetwork.size() - 1)
        sortedNetwork[i]->Forward(layerInputs[i-1], layerOutputs[i]);
      else
        sortedNetwork[i]->Forward(layerInputs[i-1], networkOutput);
    }
  }
  else if (sortedNetwork.size() == 1)
  {
    sortedNetwork[0]->Forward(input, networkOutput);
  }
  else
  {
    throw std::invalid_argument("DAGNetwork::Forward(): cannot use network with no "
        "layers!");
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
  CheckNetwork("DAGNetwork::Loss()", 0);
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
  results.set_size(sortedNetwork.back()->OutputSize(), predictors.n_cols);
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
