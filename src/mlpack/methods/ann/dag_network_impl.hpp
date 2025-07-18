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
    parentsList(network.parentsList),
    childrenList(network.childrenList),
    layerAxes(network.layerAxes),
    sortedNetwork(network.sortedNetwork),
    sortedIndices(network.sortedIndices),
    predictors(network.predictors),
    responses(network.responses),
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
    parentsList(std::move(network.parentsList)),
    childrenList(std::move(network.childrenList)),
    layerAxes(std::move(network.layerAxes)),
    sortedNetwork(std::move(network.sortedNetwork)),
    sortedIndices(std::move(network.sortedIndices)),
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
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
    childrenList = other.childrenList;
    parentsList = other.parentsList;
    layerAxes = other.layerAxes;
    sortedNetwork = other.sortedNetwork;
    sortedIndices = other.sortedIndices;
    networkOutput = other.networkOutput;
    predictors = other.predictors;
    responses = other.responses;

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
    childrenList = std::move(other.childrenList);
    parentsList = std::move(other.parentsList);
    layerAxes = std::move(other.layerAxes);
    sortedNetwork = std::move(other.sortedNetwork);
    sortedIndices = std::move(other.sortedIndices);
    networkOutput = std::move(other.networkOutput);
    predictors = std::move(other.predictors);
    responses = std::move(other.responses);

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
    std::ostringstream errorMessage;
    errorMessage << "DAGNetwork::SetAxis(): layer " << layerId << " does not exist in the network.";
    throw std::logic_error(errorMessage.str());
  }

  layerAxes[network[layerId]] = concatAxis;

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
>::Connect(size_t parentNodeId, size_t childNodeId)
{
  if (parentNodeId == childNodeId)
  {
    throw std::logic_error("DAGNetwork::Connect(): `parentNodeId` "
      "and `childNodeId` cannot be the same.");
  }

  if (parentNodeId >= network.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "DAGNetwork::Connect(): Layer "
      << parentNodeId
      << " must exist in the network before using `Connect`.";
    throw std::logic_error(errorMessage.str());
  }

  if (childNodeId >= network.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "DAGNetwork::Connect(): Layer "
      << childNodeId
      << " must exist in the network before using `Connect`.";
    throw std::logic_error(errorMessage.str());
  }

  std::vector<Layer<MatType>*>& childNodeParents = parentsList[network[childNodeId]];
  for (size_t i = 0; i < childNodeParents.size(); i++)
  {
    if (childNodeParents[i] == network[parentNodeId])
    {
      std::ostringstream errorMessage;
      errorMessage << "DAGNetwork::Connect(): Layer "
        << parentNodeId
        << " cannot be concatenated with itself.";
      throw std::logic_error(errorMessage.str());
    }
  }
  childNodeParents.push_back(network[parentNodeId]);
  childrenList[network[parentNodeId]].push_back(network[childNodeId]);

  validOutputDimensions = false;
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
  // Check that the graph has only one input and one output
  size_t inputLayers = 0;
  size_t outputLayers = 0;
  size_t outputLayerId = 0;
  for (size_t i = 0; i < network.size(); i++)
  {
    if (parentsList[network[i]].size() == 0)
    {
      inputLayers++; 
    }
    if (childrenList[network[i]].size() == 0)
    {
      outputLayers++;
      outputLayerId = i;
    }
  }

  if (inputLayers > 1)
  {
    std::ostringstream errorMessage;
    errorMessage << "DAGNetwork::CheckGraph(): There should only be one input node, "
                    "but this network has " << inputLayers << " input nodes.";
    throw std::logic_error(errorMessage.str());
  }
  if (outputLayers > 1)
  {
    std::ostringstream errorMessage;
    errorMessage << "DAGNetwork::CheckGraph(): There should only be one output node, "
                    "but this network has " << outputLayers << " output nodes.";
    throw std::logic_error(errorMessage.str());
  }

  /*
    Topological sort using an iterative depth-first search approach.

    We search from the output node through it's parents, until
    we reach a node without any parents (i.e the input layer, where there can
    only be one input layer). We maintain a stack of pairs (node, bool)
    tracking whether or not a nodes parents (and grand-parents, etc)
    have been explored.

    If a nodes parents, grand-parents etc have been explored, we add this node
    to `exploredLayers`, and push it onto `sortedNetwork`, which is equivalent to
    `network` but topologically sorted.

    If a nodes parents, grand-parents etc have not been explored, we check that
    no edge has already been traversed. If an edge has already been traversed,
    (i.e a node is it's own parent) we have found a cycle. Otherwise we add
    the pair (node, true) to the stack, indicating it's parents, grand-parents etc
    have been explored. The current nodes parents are then added to the stack as
    (parent[i], false), indicating that the parent has parents, grand-parents etc
    that have not been searched.
  */
  sortedNetwork.clear();

  std::unordered_set<Layer<MatType>*> exploredLayers;
  std::vector<std::pair<Layer<MatType>*, bool>> exploreNext;
  exploreNext.push_back(std::make_pair(network[outputLayerId], false));

  using LayerEdge = std::pair<Layer<MatType>*, Layer<MatType>*>;
  std::vector<LayerEdge> layerEdges;

  while (!exploreNext.empty())
  {
    auto [currentLayer, explored] = exploreNext.back();
    exploreNext.pop_back();

    if (exploredLayers.count(currentLayer))
      continue;

    if (!explored)
    {
      const std::vector<Layer<MatType>*>& parents = parentsList[currentLayer];

      // If an edge has already been traversed, there is a cycle.
      for (size_t i = 0; i < parents.size(); i++)
      {
        LayerEdge edge = std::make_pair(parents[i], currentLayer);
        for (size_t j = 0; j < layerEdges.size(); j++)
        {
          if (layerEdges[j] == edge)
            throw std::logic_error("DAGNetwork::CheckGraph(): A cycle "
              "exists in the graph.");
        }
        layerEdges.push_back(edge);
      }

      exploreNext.push_back({ currentLayer, true });
      for (size_t i = 0; i < parents.size(); i++)
      {
        exploreNext.push_back({ parents[i], false });
      }
    }
    else
    {
      exploredLayers.insert(currentLayer);
      sortedNetwork.push_back(currentLayer);
    }
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

  // `CheckGraph` guarantees that the first layer in `sortedNetwork`
  // has no parents and that there is only one layer with no parents.
  Layer<MatType>* currentLayer = sortedNetwork[0];
  currentLayer->InputDimensions() = inputDimensions;
  currentLayer->ComputeOutputDimensions();

  for (size_t i = 1; i < sortedNetwork.size(); i++)
  {
    currentLayer = sortedNetwork[i];
    size_t numParents = parentsList[currentLayer].size();
    if (numParents == 1)
    {
      currentLayer->InputDimensions() =
        parentsList[currentLayer][0]->OutputDimensions();
    }
    else
    {
      if (layerAxes.count(currentLayer) == 0)
      {
        std::ostringstream errorMessage;
        errorMessage << "DAGNetwork::ComputeOutputDimensions(): "
                        "Axis does not exist for layer "
                     << FindLayerIndex(currentLayer) << ".";
        throw std::logic_error(errorMessage.str());
      }

      size_t axis = layerAxes[currentLayer];
      const size_t numOutputDimensions =
        parentsList[currentLayer][0]->OutputDimensions().size();

      if (axis >= numOutputDimensions)
      {
        std::ostringstream errorMessage;
        errorMessage << "DAGNetwork::ComputeOutputDimensions(): "
                        "The concatenation axis of layer "
                     << FindLayerIndex(currentLayer) << " is " << axis
                     << ", but that's greater than or equal to the number "
                        "of output dimensions, which is "
                     << numOutputDimensions << ".";
        throw std::logic_error(errorMessage.str());
      }

      for (size_t i = 1; i < numParents; i++)
      {
        Layer<MatType>* parent = parentsList[currentLayer][i];

        if (numOutputDimensions != parent->OutputDimensions().size())
        {
          std::ostringstream errorMessage;
          errorMessage << "DAGNetwork::ComputeOutputDimensions(): "
                          "Number of output dimensions for layer 0 ("
                       << numOutputDimensions << ") should be equal "
                          "to the number of output dimensions for layer "
                       << FindLayerIndex(parent)
                       << " (" << parent->OutputDimensions().size() << ").";
          throw std::logic_error(errorMessage.str());
        }
      }
      currentLayer->InputDimensions()
        = std::vector<size_t>(numOutputDimensions, 0);

      for (size_t i = 0; i < currentLayer->InputDimensions().size(); i++)
      {
        if (i == axis)
        {
          for (size_t n = 0; n < numParents; n++)
          {
            Layer<MatType>* parent = parentsList[currentLayer][n];
            currentLayer->InputDimensions()[i] += parent->OutputDimensions()[i];
          }
        }
        else
        {
          // Set dimension not along concatenation axis to axisDim
          // as long as these dimensions are equal along all input.
          Layer<MatType>* firstParent = parentsList[currentLayer].front();
          const size_t axisDim = firstParent->OutputDimensions()[i];
          for (size_t n = 1; n < parentsList[currentLayer].size(); n++)
          {
            Layer<MatType>* parent = parentsList[currentLayer][n];
            const size_t axisDim2 = parent->OutputDimensions()[i];

            if (axisDim != axisDim2)
            {
              std::ostringstream errorMessage;
              errorMessage << "DAGNetwork::ComputeOutputDimensions(): "
                              "Axes not on the concatenation axis (axis = "
                            << axis << ") should be equal, but "
                            << axisDim << " != "
                            << axisDim2 << " along axis " << i << ".";
              throw std::logic_error(errorMessage.str());
            }
          }
          currentLayer->InputDimensions()[i] = axisDim;
        }
      }
    }
    currentLayer->ComputeOutputDimensions();
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
    throw std::logic_error(functionName + ": input size does not match "
      "expected size set with InputDimensions().");
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
>::Shuffle()
{
  ShuffleData(predictors, responses, predictors, responses);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::ResetData(MatType predictors, MatType responses)
{
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  // Set the network to training mode.
  SetNetworkMode(true);
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
    throw std::invalid_argument(functionName + ": Cannot use a network "
        "without any layers!");
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

  size_t layerOutputSize = 0;
  for (size_t i = 0; i < sortedNetwork.size() - 1; i++)
  {
    layerOutputSize += sortedNetwork[i]->OutputSize();
  }

  size_t totalConcatSize = 0;
  for (size_t i = 1; i < sortedNetwork.size(); i++)
  {
    if (parentsList[sortedNetwork[i]].size() > 1)
    {
      size_t concatSize = sortedNetwork[i]->InputDimensions()[0];
      for (size_t j = 1; j < sortedNetwork[i]->InputDimensions().size(); j++)
      {
        concatSize *= sortedNetwork[i]->InputDimensions()[j];
      }
      totalConcatSize += concatSize;
    }
  }

  residualMemorySize = totalConcatSize + layerOutputSize;
  if (batchSize * residualMemorySize > layerOutputMatrix.n_elem)
  {
    layerOutputMatrix = MatType(1, batchSize * residualMemorySize);
  }

  size_t offset = 0;
  sortedIndices.clear();

  // The first section of layerOuputMatrix (layerOutputSize)
  // gets used for layerOutputs.
  // layerInputs will be aliases to layerOutputs unless
  // those layers have multiple parents. If thats the case
  // those layerInputs will aliases to the second section
  // of layerOutputMatrix (totalConcatSize).

  //setup layerOutputs
  for (size_t i = 0; i < sortedNetwork.size() - 1; i++)
  {
    const size_t layerOutputSize = sortedNetwork[i]->OutputSize();
    MakeAlias(layerOutputs[i], layerOutputMatrix, layerOutputSize,
      batchSize, offset);
    offset += batchSize * layerOutputSize;
    sortedIndices.insert({sortedNetwork[i], i});
  }

  //setup layerInputs
  for (size_t i = 1; i < sortedNetwork.size(); i++)
  {
    Layer<MatType>* currentLayer = sortedNetwork[i];
    const std::vector<Layer<MatType>*>& parents = parentsList[currentLayer];
    size_t numParents = parents.size();

    if (numParents == 1)
    {
      size_t inputIndex = sortedIndices[parents.front()];
      MakeAlias(layerInputs[i - 1], layerOutputs[inputIndex], 
        layerOutputs[inputIndex].n_rows, layerOutputs[inputIndex].n_cols, 0);
    }
    else
    {
      size_t concatSize = currentLayer->InputDimensions()[0];
      for (size_t j = 1; j < currentLayer->InputDimensions().size(); j++)
        concatSize *= currentLayer->InputDimensions()[j];
      MakeAlias(layerInputs[i - 1], layerOutputMatrix, concatSize,
        batchSize, offset);
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
>::InitializeBackwardPassMemory(const size_t batchSize)
{
  CheckNetwork("DAGNetwork::InitializeBackwardPassMemory()", 0);

  if (network.size() <= 1)
    return;

  size_t deltaMatrixSize = 0;
  for (size_t i = 0; i < sortedNetwork.size() - 1; i++)
  {
    Layer<MatType>* currentLayer = sortedNetwork[i];

    size_t layerDeltaSize = currentLayer->OutputSize();
    size_t concatDeltaSize = 0;

    if (childrenList[currentLayer].size() > 1)
      layerDeltaSize *= 2;

    if (parentsList[currentLayer].size() > 1)
    {
      concatDeltaSize = currentLayer->InputDimensions()[0];
      for (size_t j = 1; j < currentLayer->InputDimensions().size(); j++)
        concatDeltaSize *= currentLayer->InputDimensions()[j];
    }
    deltaMatrixSize += layerDeltaSize + concatDeltaSize;
  }

  if (batchSize * deltaMatrixSize > layerDeltaMatrix.n_elem)
    layerDeltaMatrix = MatType(1, batchSize * deltaMatrixSize);

  size_t offset = 0;
  for (size_t i = 0; i < sortedNetwork.size() - 1; i++)
  {
    Layer<MatType>* currentLayer = sortedNetwork[i];
    if (childrenList[currentLayer].size() > 1)
    {
      accumulatedDeltas.insert({currentLayer, layerDeltas[i]});
      MakeAlias(accumulatedDeltas[currentLayer], layerDeltaMatrix,
        currentLayer->OutputSize(), batchSize, offset);
      offset += currentLayer->OutputSize() * batchSize;
      outputDeltas.insert({currentLayer, MatType()});
    }
    else
       outputDeltas.insert({currentLayer, layerDeltas[i]});

    MakeAlias(outputDeltas[currentLayer], layerDeltaMatrix,
       currentLayer->OutputSize(), batchSize, offset);
    offset += currentLayer->OutputSize() * batchSize;
  }

  for (size_t i = 1; i < sortedNetwork.size(); i++)
  {
    Layer<MatType>* currentLayer = sortedNetwork[i];
    std::vector<Layer<MatType>*> parents = parentsList[currentLayer]; 
    if (parents.size() > 1)
    {
      inputDeltas.insert({currentLayer, MatType()});

      size_t inputSize = currentLayer->InputDimensions()[0];
      for (size_t j = 1; j < currentLayer->InputDimensions().size(); j++)
        inputSize *= currentLayer->InputDimensions()[j];

      MakeAlias(inputDeltas[currentLayer], layerDeltaMatrix,
        inputSize, batchSize, offset);
      offset += inputSize * batchSize;
    }
    else
    {
      Layer<MatType>* onlyParent = parents.front();
      inputDeltas[currentLayer] = outputDeltas[onlyParent];
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
      const std::vector<Layer<MatType>*>& parents = 
        parentsList[sortedNetwork[i]];

      if (parents.size() > 1)
      {
        size_t axis = layerAxes[sortedNetwork[i]];

        size_t rows = 1;
        for (size_t j = 0; j < axis; j++)
          rows *= sortedNetwork[i]->InputDimensions()[j];

        size_t slices = input.n_cols;
        size_t numDims = sortedNetwork[i]->InputDimensions().size();
        for (size_t j = axis + 1; j < numDims; j++)
          slices *= sortedNetwork[i]->InputDimensions()[j];

        std::vector<CubeType> parentOutputAliases(parents.size());
        for (size_t j = 0; j < parents.size(); j++)
        {
          size_t cols = parents[j]->OutputDimensions()[axis];
          MatType& parentOutput = layerOutputs[sortedIndices[parents[j]]];
          MakeAlias(parentOutputAliases[j], parentOutput, rows, cols, slices);
        }

        CubeType inputAlias;
        MakeAlias(inputAlias, layerInputs[i-1], rows,
          sortedNetwork[i]->InputDimensions()[axis], slices);

        size_t startCol = 0;
        for (size_t j = 0; j < parentOutputAliases.size(); j++)
        {
          const size_t cols = parentOutputAliases[j].n_cols;
          inputAlias.cols(startCol, startCol + cols - 1) = parentOutputAliases[j];
          startCol += cols;
        }
      }

      // Don't execute if it's the last iteration.
      if (i < sortedNetwork.size() - 1)
        sortedNetwork[i]->Forward(layerInputs[i-1], layerOutputs[i]);
    }
    sortedNetwork.back()->Forward(layerInputs.back(), networkOutput);
  }
  else if (sortedNetwork.size() == 1)
  {
    sortedNetwork[0]->Forward(input, networkOutput);
  }
  else
  {
    throw std::invalid_argument("DAGNetwork::Forward(): Cannot use network"
        " with no layers!");
  }

  if (&results != &networkOutput)
    results = networkOutput;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Backward(const MatType& input,
            const MatType& output,
            const MatType& error,
            MatType& gradients)
{
  CheckNetwork("DAGNetwork::Backward()", input.n_rows);
  gradients.set_size(parameters.n_rows, parameters.n_cols);

  if (network.size() > 1)
  {
    InitializeBackwardPassMemory(input.n_cols);
  }
  else if (network.size() == 1)
  {
    network[0]->Gradient(input, error, gradients);
  }
  else
  {
    // Nothing to do if the network is empty... there is no gradient.
  }
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

// template<typename OutputLayerType,
//          typename InitializationRuleType,
//          typename MatType>
// typename MatType::elem_type DAGNetwork<
//     OutputLayerType,
//     InitializationRuleType,
//     MatType
// >::Evaluate(const MatType& parameters)
// {
//   typename MatType::elem_type res = 0;
//   for (size_t i = 0; i < predictors.n_cols; ++i)
//     res += Evaluate(parameters, i, 1);
//
//   return res;
// }

// template<typename OutputLayerType,
//          typename InitializationRuleType,
//          typename MatType>
// typename MatType::elem_type DAGNetwork<
//     OutputLayerType,
//     InitializationRuleType,
//     MatType
// >::Evaluate(const MatType& predictors, const MatType& responses)
// {
//   CheckNetwork("DAGNetwork::Evaluate()", predictors.n_rows);
//   Forward(predictors, networkOutput);
//   return outputLayer.Forward(networkOutput, responses) + Loss();
// }

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

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename OptimizerType>
std::enable_if_t<
    ens::traits::HasMaxIterationsSignature<OptimizerType>::value, void>
DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const
{
  if (optimizer.MaxIterations() < samples &&
      optimizer.MaxIterations() != 0)
  {
    Log::Warn << "The optimizer's maximum number of iterations is less than the"
        << " size of the dataset; the optimizer will not pass over the entire "
        << "dataset. To fix this, modify the maximum number of iterations to be"
        << " at least equal to the number of points of your dataset ("
        << samples << ")." << std::endl;
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename OptimizerType>
std::enable_if_t<
    !ens::traits::HasMaxIterationsSignature<OptimizerType>::value, void>
DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::WarnMessageMaxIterations(OptimizerType& /* optimizer */,
                            size_t /* samples */) const
{
  // Nothing to do here.
}

} // namespace mlpack

#endif
