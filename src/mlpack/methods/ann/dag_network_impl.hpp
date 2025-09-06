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
    graphIsSet(false),
    layerMemoryIsSet(false)
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
>::DAGNetwork(const DAGNetwork& other) :
    outputLayer(other.outputLayer),
    initializeRule(other.initializeRule),

    childrenList(other.childrenList),
    parentsList(other.parentsList),
    layerAxes(other.layerAxes),

    parameters(other.parameters),
    inputDimensions(other.inputDimensions),
    predictors(other.predictors),
    responses(other.responses),

    validOutputDimensions(false),
    graphIsSet(false),
    layerMemoryIsSet(false)
{
  for (size_t i = 0; i < other.network.size(); i++)
    network.push_back(other.network[i]->Clone());

  size_t size = std::max<int>(network.size() - 1, 0);
  layerOutputs.resize(size, MatType());
  layerDeltas.resize(size, MatType());
  layerInputs.resize(size, MatType());

  layerGradients.resize(network.size(), MatType());
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::DAGNetwork(DAGNetwork&& other) :
    outputLayer(std::move(other.outputLayer)),
    initializeRule(std::move(other.initializeRule)),

    network(std::move(other.network)),
    layerAxes(std::move(other.layerAxes)),

    parameters(std::move(other.parameters)),
    inputDimensions(std::move(other.inputDimensions)),
    predictors(std::move(other.predictors)),
    responses(std::move(other.responses)),

    validOutputDimensions(false),
    graphIsSet(false),
    layerMemoryIsSet(false)
{
  size_t size = std::max<int>(network.size() - 1, 0);
  layerOutputs.resize(size, MatType());
  layerDeltas.resize(size, MatType());
  layerInputs.resize(size, MatType());

  layerGradients.resize(network.size(), MatType());

  parentsList = std::move(other.parentsList);
  childrenList = std::move(other.childrenList);
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

    parentsList = other.parentsList;
    childrenList = other.childrenList;
    layerAxes = other.layerAxes;

    parameters = other.parameters;
    inputDimensions = other.inputDimensions;
    predictors = other.predictors;
    responses = other.responses;

    validOutputDimensions = false;
    graphIsSet = false;
    layerMemoryIsSet = false;

    for (size_t i = 0; i < other.network.size(); i++)
      network.push_back(other.network[i]->Clone());

    size_t size = std::max<int>(network.size() - 1, 0);
    layerOutputs.resize(size, MatType());
    layerDeltas.resize(size, MatType());
    layerInputs.resize(size, MatType());

    layerGradients.resize(network.size(), MatType());
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
    parentsList = std::move(other.parentsList);
    childrenList = std::move(other.childrenList);
    layerAxes = std::move(other.layerAxes);

    parameters = std::move(other.parameters);
    inputDimensions = std::move(other.inputDimensions);
    predictors = std::move(other.predictors);
    responses = std::move(other.responses);

    validOutputDimensions = false;
    graphIsSet = false;
    layerMemoryIsSet = false;

    size_t size = std::max<int>(network.size() - 1, 0);
    layerOutputs.resize(size, MatType());
    layerDeltas.resize(size, MatType());
    layerInputs.resize(size, MatType());

    layerGradients.resize(network.size(), MatType());
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
    errorMessage << "DAGNetwork::SetAxis(): layer "
      << layerId << " does not exist in the network.";
    throw std::logic_error(errorMessage.str());
  }

  layerAxes[layerId] = concatAxis;

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
    throw std::logic_error("DAGNetwork::Connect(): `parentNodeId` "
      "and `childNodeId` cannot be the same.");

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

  std::vector<size_t>& childNodeParents = parentsList[childNodeId];
  for (size_t i = 0; i < childNodeParents.size(); i++)
  {
    if (childNodeParents[i] == parentNodeId)
    {
      std::ostringstream errorMessage;
      errorMessage << "DAGNetwork::Connect(): Layer "
        << parentNodeId
        << " cannot be concatenated with itself.";
      throw std::logic_error(errorMessage.str());
    }
  }
  childNodeParents.push_back(parentNodeId);
  childrenList[parentNodeId].push_back(childNodeId);

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
    if (parentsList[i].size() == 0)
      inputLayers++;

    if (childrenList[i].size() == 0)
    {
      outputLayers++;
      outputLayerId = i;
    }
  }

  if (inputLayers == 0 || outputLayers == 0)
    throw std::logic_error("DAGNetwork::CheckGraph(): A cycle "
      "exists in the graph.");

  if (inputLayers > 1)
  {
    std::ostringstream errorMessage;
    errorMessage << "DAGNetwork::CheckGraph(): "
      "There should be exactly one input node, "
      "but this network has " << inputLayers << " input nodes.";
    throw std::logic_error(errorMessage.str());
  }
  if (outputLayers > 1)
  {
    std::ostringstream errorMessage;
    errorMessage << "DAGNetwork::CheckGraph(): "
      "There should be exactly one output node, "
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
    to `exploredLayers`, and push it onto `sortedNetwork`, which is equivalent
    to `network` but topologically sorted.

    If a nodes parents, grand-parents etc have not been explored, we check that
    no edge has already been traversed. If an edge has already been traversed,
    (i.e a node is it's own parent) we have found a cycle. Otherwise we add
    the pair (node, true) to the stack, indicating it's parents, grand-parents
    etc have been explored. The current nodes parents are then added to the
    stack as (parent[i], false), indicating that the parent has parents,
    grand-parents etc that have not been searched.
  */
  sortedNetwork.clear();

  std::unordered_set<size_t> exploredLayers;
  std::vector<std::pair<size_t, bool>> exploreNext;
  exploreNext.push_back({ outputLayerId, false });

  using LayerEdge = std::pair<size_t, size_t>;
  std::vector<LayerEdge> layerEdges;

  while (!exploreNext.empty())
  {
    auto [layer, explored] = exploreNext.back();
    exploreNext.pop_back();

    if (exploredLayers.count(layer))
      continue;

    if (!explored)
    {
      const std::vector<size_t>& parents = parentsList[layer];

      // If an edge has already been traversed, there is a cycle.
      for (size_t i = 0; i < parents.size(); i++)
      {
        LayerEdge edge = { parents[i], layer };
        for (size_t j = 0; j < layerEdges.size(); j++)
        {
          if (layerEdges[j] == edge)
            throw std::logic_error("DAGNetwork::CheckGraph(): A cycle "
              "exists in the graph.");
        }
        layerEdges.push_back(edge);
      }

      exploreNext.push_back({ layer, true });
      for (size_t i = 0; i < parents.size(); i++)
        exploreNext.push_back({ parents[i], false });
    }
    else
    {
      exploredLayers.insert(layer);
      sortedNetwork.push_back(layer);
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
  // has no parents and that there is only one layer with no parents,
  // which will be the last layer in `sortedNetwork`.
  size_t currentLayer = sortedNetwork[0];

  network[currentLayer]->InputDimensions() = inputDimensions;
  network[currentLayer]->ComputeOutputDimensions();

  for (size_t layer = 1; layer < sortedNetwork.size(); layer++)
  {
    currentLayer = sortedNetwork[layer];
    const size_t numParents = parentsList[currentLayer].size();
    const size_t firstParent = parentsList[currentLayer].front();
    if (numParents == 1)
    {
      network[currentLayer]->InputDimensions() =
        network[firstParent]->OutputDimensions();
    }
    else
    {
      // numParents guaranteed to be > 1.

      const size_t numOutputDimensions =
        network[firstParent]->OutputDimensions().size();

      for (size_t i = 1; i < numParents; i++)
      {
        size_t parentIndex = parentsList[currentLayer][i];
        size_t parentDims = network[parentIndex]->OutputDimensions().size();
        if (numOutputDimensions != parentDims)
        {
          std::ostringstream errorMessage;
          errorMessage << "DAGNetwork::ComputeOutputDimensions(): "
                          "Number of output dimensions for layer 0 ("
                       << numOutputDimensions << ") should be equal "
                          "to the number of output dimensions for layer "
                       << parentIndex << ", which is " << parentDims << ".";
          throw std::logic_error(errorMessage.str());
        }
      }

      // If layerAxes for currentLayer is not set, set it to the last dimension
      // by default.
      if (layerAxes.count(currentLayer) == 0)
        layerAxes.insert({ currentLayer,
          network[firstParent]->OutputDimensions().size() - 1 });

      const size_t axis = layerAxes[currentLayer];
      if (axis >= numOutputDimensions)
      {
        std::ostringstream errorMessage;
        errorMessage << "DAGNetwork::ComputeOutputDimensions(): "
                        "The concatenation axis of layer "
                     << currentLayer << " is " << axis
                     << ", but that's greater than or equal to the number "
                        "of output dimensions, which is "
                     << numOutputDimensions << ".";
        throw std::logic_error(errorMessage.str());
      }

      network[currentLayer]->InputDimensions() =
          std::vector<size_t>(numOutputDimensions, 0);

      for (size_t i = 0; i < numOutputDimensions; i++)
      {
        if (i == axis)
        {
          for (size_t j = 0; j < numParents; j++)
          {
            const size_t parentIndex = parentsList[currentLayer][j];
            network[currentLayer]->InputDimensions()[i] +=
                network[parentIndex]->OutputDimensions()[i];
          }
        }
        else
        {
          // Set dimensions not along the concatenation axis to axisDim
          // as long as these dimensions are all equal.
          const size_t axisDim = network[firstParent]->OutputDimensions()[i];
          for (size_t j = 1; j < parentsList[currentLayer].size(); j++)
          {
            const size_t parentIndex = parentsList[currentLayer][j];
            Layer<MatType>* parent = network[parentIndex];
            const size_t currentAxisDim = parent->OutputDimensions()[i];

            if (axisDim != currentAxisDim)
            {
              std::ostringstream errorMessage;
              errorMessage << "DAGNetwork::ComputeOutputDimensions(): "
                              "Axes not on the concatenation axis (axis = "
                            << axis << ") should be equal, but "
                            << axisDim << " != "
                            << currentAxisDim << " along axis " << i
                            << " for layers " << network[firstParent]
                            << " and " << network[parentIndex] << ".";
              throw std::logic_error(errorMessage.str());
            }
          }
          network[currentLayer]->InputDimensions()[i] = axisDim;
        }
      }

      rowsCache.clear();
      size_t rows = 1;
      for (size_t j = 0; j < axis; j++)
        rows *= network[currentLayer]->InputDimensions()[j];
      rowsCache.insert({ currentLayer, rows });

      slicesCache.clear();
      size_t slices = 1;
      for (size_t j = axis + 1; j < numOutputDimensions; j++)
        slices *= network[currentLayer]->InputDimensions()[j];
      slicesCache.insert({ currentLayer, slices });
    }
    network[currentLayer]->ComputeOutputDimensions();
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
    throw std::logic_error(functionName + ": Input size does not match "
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
    throw std::invalid_argument("DAGNetwork::Reset(): Cannot reset network when"
        "no input dimensionality is given, and `InputDimensions()` has not been"
        " set!");
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
  size_t offset = 0;
  const size_t totalWeightSize = WeightSize();
  for (size_t i = 0; i < network.size(); i++)
  {
    const size_t weightSize = network[i]->WeightSize();

    if (offset + weightSize > totalWeightSize)
    {
      throw std::logic_error("DAGNetwork::SetLayerMemory(): Parameter "
        "size does not match total layer weight size!");
    }
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
  if (!graphIsSet)
    CheckGraph();

  size_t offset = 0;
  const size_t totalWeightSize = elements;
  for (size_t i = 0; i < sortedNetwork.size(); ++i)
  {
    const size_t index = sortedNetwork[i];
    const size_t weightSize = network[index]->WeightSize();
    if (offset + weightSize > totalWeightSize)
    {
      throw std::logic_error("DAGNetwork::SetLayerMemory(): Parameter "
        "size does not match total layer weight size!");
    }

    MatType WTemp;
    MakeAlias(WTemp, W, weightSize, 1, offset);
    network[index]->CustomInitialize(WTemp, weightSize);

    offset += weightSize;
  }

  if (offset != totalWeightSize)
  {
    throw std::logic_error("DAGNetwork::CustomInitialize(): Total layer "
      "weight size does not match rows size!");
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
  networkInit.Initialize(SortedNetwork(), parameters);
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
    throw std::logic_error("DAGNetwork::SetLayerMemory(): Total layer weight "
      "size does not match parameter size!");
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

  if (parameters.n_elem != WeightSize())
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
    const size_t currentLayer = sortedNetwork[i];
    layerOutputSize += network[currentLayer]->OutputSize();
  }

  size_t totalConcatSize = 0;
  for (size_t i = 1; i < sortedNetwork.size(); i++)
  {
    const size_t currentLayer = sortedNetwork[i];
    Layer<MatType>* layer = network[currentLayer];
    if (parentsList[currentLayer].size() > 1)
    {
      size_t concatSize = layer->InputDimensions()[0];
      for (size_t j = 1; j < layer->InputDimensions().size(); j++)
        concatSize *= layer->InputDimensions()[j];

      totalConcatSize += concatSize;
    }
  }

  const size_t activationMemorySize = totalConcatSize + layerOutputSize;
  if (batchSize * activationMemorySize == layerOutputMatrix.n_elem)
    return;

  layerOutputMatrix = MatType(1, batchSize * activationMemorySize);

  size_t offset = 0;
  sortedIndices.clear();

  // The first section of layerOuputMatrix (layerOutputSize)
  // gets used for layerOutputs.
  // layerInputs will be aliases to layerOutputs unless
  // those layers have multiple parents. It's inputs will need
  // to be concatenated first). If thats the case
  // those layerInputs will aliases to the second section
  // of layerOutputMatrix (totalConcatSize).

  //setup layerOutputs
  for (size_t i = 0; i < sortedNetwork.size() - 1; i++)
  {
    const size_t currentLayer = sortedNetwork[i];
    const size_t layerOutputSize = network[currentLayer]->OutputSize();
    MakeAlias(layerOutputs[i], layerOutputMatrix, layerOutputSize,
      batchSize, offset);
    offset += batchSize * layerOutputSize;
    sortedIndices.insert({ sortedNetwork[i], i });
  }

  sortedIndices.insert({
    sortedNetwork.back(), sortedNetwork.size() - 1
  });

  //setup layerInputs
  for (size_t i = 1; i < sortedNetwork.size(); i++)
  {
    const size_t currentLayer = sortedNetwork[i];
    Layer<MatType>* layer = network[currentLayer];
    const std::vector<size_t>& parents = parentsList[currentLayer];
    const size_t numParents = parents.size();

    if (numParents == 1)
    {
      const size_t parentIndex = sortedIndices[parents.front()];
      MakeAlias(layerInputs[i - 1], layerOutputs[parentIndex],
        layerOutputs[parentIndex].n_rows, layerOutputs[parentIndex].n_cols, 0);
    }
    else
    {
      size_t concatSize = layer->InputDimensions()[0];
      for (size_t j = 1; j < layer->InputDimensions().size(); j++)
        concatSize *= layer->InputDimensions()[j];
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
  for (size_t i = 0; i < sortedNetwork.size(); i++)
  {
    const size_t currentLayer = sortedNetwork[i];
    Layer<MatType>* layer = network[currentLayer];
    size_t layerDeltaSize = layer->OutputSize();
    size_t concatDeltaSize = 0;

    if (childrenList[currentLayer].size() > 1)
      layerDeltaSize *= 2;

    if (parentsList[currentLayer].size() > 1)
    {
      concatDeltaSize = layer->InputDimensions()[0];
      for (size_t j = 1; j < layer->InputDimensions().size(); j++)
        concatDeltaSize *= layer->InputDimensions()[j];
    }

    // If at the last layer, we only need to add it's `concatDeltaSize`.
    if (i == sortedNetwork.size() - 1)
      layerDeltaSize = 0;

    deltaMatrixSize += layerDeltaSize + concatDeltaSize;
  }

  if (batchSize * deltaMatrixSize == layerDeltaMatrix.n_elem)
    return;

  layerDeltaMatrix = MatType(1, batchSize * deltaMatrixSize);
  outputDeltas.clear();
  accumulatedDeltas.clear();
  inputDeltas.clear();

  size_t offset = 0;
  for (size_t i = 0; i < sortedNetwork.size() - 1; i++)
  {
    const size_t currentLayer = sortedNetwork[i];

    outputDeltas.insert({ i, MatType() });
    MakeAlias(outputDeltas[i], layerDeltaMatrix,
       network[currentLayer]->OutputSize(), batchSize, offset);
    offset += network[currentLayer]->OutputSize() * batchSize;

    if (childrenList[currentLayer].size() > 1)
    {
      accumulatedDeltas.insert({ i, MatType() });

      MakeAlias(accumulatedDeltas[i], layerDeltaMatrix,
        network[currentLayer]->OutputSize(), batchSize, offset);
      MakeAlias(layerDeltas[i], accumulatedDeltas[i],
        accumulatedDeltas[i].n_rows, accumulatedDeltas[i].n_cols);

      offset += network[currentLayer]->OutputSize() * batchSize;
    }
    else
    {
      MakeAlias(layerDeltas[i], outputDeltas[i], outputDeltas[i].n_rows,
        outputDeltas[i].n_cols);
    }
  }

  for (size_t i = 1; i < sortedNetwork.size(); i++)
  {
    const size_t currentLayer = sortedNetwork[i];
    Layer<MatType>* layer = network[currentLayer];

    const std::vector<size_t>& parents = parentsList[currentLayer];
    inputDeltas.insert({ i, MatType() });

    if (parents.size() > 1)
    {
      size_t inputSize = layer->InputDimensions()[0];
      for (size_t j = 1; j < layer->InputDimensions().size(); j++)
        inputSize *= layer->InputDimensions()[j];

      MakeAlias(inputDeltas[i], layerDeltaMatrix,
        inputSize, batchSize, offset);
      offset += inputSize * batchSize;
    }
    else
    {
      size_t onlyParent = sortedIndices[parents.front()];
      MakeAlias(inputDeltas[i], outputDeltas[onlyParent],
        outputDeltas[onlyParent].n_rows, outputDeltas[onlyParent].n_cols);
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
>::InitializeGradientPassMemory(MatType& gradient)
{
  size_t gradientStart = 0;
  for (size_t i = 0; i < sortedNetwork.size(); ++i)
  {
    const size_t currentLayer = sortedNetwork[i];
    const size_t weightSize = network[currentLayer]->WeightSize();
    MakeAlias(layerGradients[i], gradient, weightSize, 1, gradientStart);
    gradientStart += weightSize;
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

  const size_t lastLayer = sortedNetwork.back();
  networkOutput.set_size(network[lastLayer]->OutputSize(), input.n_cols);

  // equivalent to MultiLayer->Forward(), but with a concat.
  if (sortedNetwork.size() > 1)
  {
    InitializeForwardPassMemory(input.n_cols);

    const size_t firstLayer = sortedNetwork.front();
    network[firstLayer]->Forward(input, layerOutputs.front());

    for (size_t i = 1; i < sortedNetwork.size(); i++)
    {
      const size_t currentLayer = sortedNetwork[i];
      const std::vector<size_t>& parents = parentsList[currentLayer];

      // Concatenation
      if (parents.size() > 1)
      {
        const size_t axis = layerAxes[currentLayer];

        size_t rows = rowsCache[currentLayer];
        size_t slices = slicesCache[currentLayer] * input.n_cols;

        CubeType inputAlias;
        MakeAlias(inputAlias, layerInputs[i - 1], rows,
          network[currentLayer]->InputDimensions()[axis], slices);
        size_t startCol = 0;

        for (size_t j = 0; j < parents.size(); j++)
        {
          const size_t index = sortedIndices[parents[j]];
          const MatType& parentOutput = layerOutputs[index];

          const size_t cols = network[parents[j]]->OutputDimensions()[axis];
          CubeType parentOutputAlias;
          MakeAlias(parentOutputAlias, parentOutput, rows, cols, slices);

          inputAlias.cols(startCol, startCol + cols - 1) = parentOutputAlias;
          startCol += cols;
        }
      }

      // Don't execute if it's the last iteration
      // network[lastLayer] might need a concatenation.
      if (i < sortedNetwork.size() - 1)
        network[currentLayer]->Forward(layerInputs[i - 1], layerOutputs[i]);
    }
    network[lastLayer]->Forward(layerInputs.back(), networkOutput);
  }
  else if (sortedNetwork.size() == 1)
  {
    network[0]->Forward(input, networkOutput);
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
  CheckNetwork("DAGNetwork::Backward()", input.n_rows, true, true);
  gradients.set_size(parameters.n_rows, parameters.n_cols);
  networkDelta.set_size(input.n_rows, input.n_cols);

  if (sortedNetwork.size() > 1)
  {
    InitializeBackwardPassMemory(input.n_cols);

    for (size_t i = 0; i < sortedNetwork.size(); i++)
    {
      const size_t currentLayer = sortedNetwork[i];
      if (childrenList[currentLayer].size() > 1)
        accumulatedDeltas[i].zeros();
    }

    MatType const* currentOutput = &output;
    MatType const* currentDelta = &error;

    for (size_t i = sortedNetwork.size() - 1; i > 0; --i)
    {
      const size_t currentLayer = sortedNetwork[i];
      Layer<MatType>* layer = network[currentLayer];
      layer->Backward(layerInputs[i - 1], *currentOutput,
        *currentDelta, inputDeltas[i]);

      const size_t numParents = parentsList[currentLayer].size();
      if (numParents > 1)
      {
        // Calculating deltas for concatenation.
        const size_t axis = layerAxes[currentLayer];

        size_t batchSize = input.n_cols;
        size_t rows = rowsCache[currentLayer];
        size_t slices = slicesCache[currentLayer] * batchSize;

        CubeType inputDeltaAlias;
        MakeAlias(inputDeltaAlias, inputDeltas[i], rows,
          layer->InputDimensions()[axis], slices);

        const std::vector<size_t>& parents = parentsList[currentLayer];
        size_t startCol = 0;
        for (size_t j = 0; j < parents.size(); j++)
        {
          Layer<MatType>* parent = network[parents[j]];
          const size_t parentIndex = sortedIndices[parents[j]];
          const size_t cols = parent->OutputDimensions()[axis];
          outputDeltas[parentIndex] = inputDeltaAlias.cols(startCol,
            startCol + cols - 1);

          outputDeltas[parentIndex].reshape(
            outputDeltas[parentIndex].n_elem / batchSize, batchSize);
          startCol += cols;
        }
      }

      /*
       * If a parent has multiple children, you need to accumulate
       * the deltas instead across all it's children in order
       * to correctly calculate that layers gradient w.r.t the
       * networks loss.
       */
      for (size_t j = 0; j < numParents; j++)
      {
        const size_t parent = parentsList[currentLayer][j];
        const size_t parentNumChildren = childrenList[parent].size();
        if (parentNumChildren > 1)
        {
          const size_t parentIndex = sortedIndices[parent];
          accumulatedDeltas[parentIndex] += outputDeltas[parent];
        }
      }

      currentOutput = &layerOutputs[i - 1];
      currentDelta = &layerDeltas[i - 1];
    }
    const size_t firstLayer = sortedNetwork.front();
    network[firstLayer]->Backward(input, *currentOutput, *currentDelta,
      networkDelta);

    InitializeGradientPassMemory(gradients);

    network[firstLayer]->Gradient(input, layerDeltas.front(),
      layerGradients.front());
    for (size_t i = 1; i < sortedNetwork.size() - 1; i++)
    {
      const size_t currentLayer = sortedNetwork[i];
      network[currentLayer]->Gradient(layerInputs[i - 1], layerDeltas[i],
        layerGradients[i]);
    }
    const size_t lastLayer = sortedNetwork.back();
    network[lastLayer]->Gradient(layerInputs.back(), error,
      layerGradients.back());
  }
  else if (sortedNetwork.size() == 1)
  {
    network[0]->Backward(input, output, error, networkDelta);
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
typename MatType::elem_type DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Evaluate(const MatType& parameters)
{
  typename MatType::elem_type res = 0;
  for (size_t i = 0; i < predictors.n_cols; i++)
    res += Evaluate(parameters, i, 1);
  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Evaluate(const MatType& /* parameters */,
            const size_t begin,
            const size_t batchSize)
{
  CheckNetwork("DAGNetwork::Evaluate()", predictors.n_rows);

  // Set networkOutput to the right size if needed, then perform the forward
  // pass.
  const size_t lastLayer = sortedNetwork.back();
  networkOutput.set_size(network[lastLayer]->OutputSize(), batchSize);
  MatType predictorsBatch, responsesBatch;
  MakeAlias(predictorsBatch, predictors, predictors.n_rows, batchSize,
      begin * predictors.n_rows);
  MakeAlias(responsesBatch, responses, responses.n_rows, batchSize,
      begin * responses.n_rows);
  Forward(predictorsBatch, networkOutput);

  return outputLayer.Forward(networkOutput, responsesBatch) + Loss();
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::EvaluateWithGradient(const MatType& parameters,
                        MatType& gradient)
{
  typename MatType::elem_type res = 0;
  res += EvaluateWithGradient(parameters, 0, gradient, 1);
  MatType tmpGradient(gradient.n_rows, gradient.n_cols,
    GetFillType<MatType>::none);
  for (size_t i = 1; i < predictors.n_cols; ++i)
  {
    res += EvaluateWithGradient(parameters, i, tmpGradient, 1);
    gradient += tmpGradient;
  }

  return res;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Gradient(const MatType& parameters,
            const size_t begin,
            MatType& gradient,
            const size_t batchSize)
{
  this->EvaluateWithGradient(parameters, begin, gradient, batchSize);
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
typename MatType::elem_type DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::EvaluateWithGradient(const MatType& parameters,
                        const size_t begin,
                        MatType& gradient,
                        const size_t batchSize)
{
  CheckNetwork("DAGNetwork::EvaluateWithGradient()", predictors.n_rows);
  const size_t lastLayer = sortedNetwork.back();
  networkOutput.set_size(network[lastLayer]->OutputSize(),
    predictors.n_cols);

  MatType predictorsBatch, responsesBatch;
  MakeAlias(predictorsBatch, predictors, predictors.n_rows,
    batchSize, begin * predictors.n_rows);
  MakeAlias(responsesBatch, responses, responses.n_rows,
    batchSize, begin * responses.n_rows);

  Forward(predictorsBatch, networkOutput);

  const typename MatType::elem_type obj =
      outputLayer.Forward(networkOutput, responsesBatch) + Loss();

  outputLayer.Backward(networkOutput, responsesBatch, error);

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  Backward(predictorsBatch, networkOutput, error, gradient);

  return obj;
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

  size_t lastLayer = sortedNetwork.back();
  results.set_size(network[lastLayer]->OutputSize(), predictors.n_cols);
  for (size_t i = 0; i < predictors.n_cols; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize,
        size_t(predictors.n_cols) - i);

    MatType predictorAlias, resultAlias;

    MakeAlias(predictorAlias, predictors, predictors.n_rows,
      effectiveBatchSize, i * predictors.n_rows);
    MakeAlias(resultAlias, results, results.n_rows,
      effectiveBatchSize, i * results.n_rows);

    Forward(predictorAlias, resultAlias);
  }
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename OptimizerType, typename... CallbackTypes>
typename MatType::elem_type DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Train(MatType predictors,
         MatType responses,
         OptimizerType& optimizer,
         CallbackTypes&&... callbacks)
{
  ResetData(std::move(predictors), std::move(responses));
  WarnMessageMaxIterations<OptimizerType>(optimizer, this->predictors.n_cols);

  CheckNetwork("DAGNetwork::Train()", this->predictors.n_rows, true, true);

  Timer::Start("dag_network_optimization");
  const typename MatType::elem_type out =
      optimizer.Optimize(*this, parameters, callbacks...);
  Timer::Stop("dag_network_optimization");

  Log::Info << "DAGNetwork::Train(): final objective of trained model is "
    << out << "." << std::endl;

  return out;
}

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename OptimizerType, typename... CallbackTypes>
typename MatType::elem_type DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::Train(MatType predictors,
         MatType responses,
         CallbackTypes&&... callbacks)
{
  OptimizerType optimizer;
  return Train(std::move(predictors), std::move(responses), optimizer,
      callbacks...);
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


template<typename OutputLayerType,
         typename InitializationRuleType,
         typename MatType>
template<typename Archive>
void DAGNetwork<
    OutputLayerType,
    InitializationRuleType,
    MatType
>::serialize(Archive& ar, const uint32_t /* version */)
{
  #if !defined(MLPACK_ENABLE_ANN_SERIALIZATION) && \
      !defined(MLPACK_ANN_IGNORE_SERIALIZATION_WARNING)
    // Note: if you define MLPACK_IGNORE_ANN_SERIALIZATION_WARNING, you had
    // better ensure that every layer you are serializing has had
    // CEREAL_REGISTER_TYPE() called somewhere.  See layer/serialization.hpp for
    // more information.
    throw std::runtime_error("Cannot serialize a neural network unless "
        "MLPACK_ENABLE_ANN_SERIALIZATION is defined!  See the \"Additional "
        "build options\" section of the README for more information.");

    (void) ar;
  #else
    // Serialize the output layer and initialization rule.
    ar(CEREAL_NVP(outputLayer));
    ar(CEREAL_NVP(initializeRule));

    // Serialize the network itself.
    ar(CEREAL_VECTOR_POINTER(network));
    ar(CEREAL_NVP(childrenList));
    ar(CEREAL_NVP(parentsList));
    ar(CEREAL_NVP(layerAxes));
    ar(CEREAL_NVP(parameters));

    ar(CEREAL_NVP(inputDimensions));

    // If we are loading, we need to initialize the weights.
    if (cereal::is_loading<Archive>())
    {
      sortedIndices.clear();

      predictors.clear();
      responses.clear();

      networkOutput.clear();
      networkDelta.clear();
      error.clear();

      size_t size = std::max<int>(network.size() - 1, 0);
      layerOutputs.resize(size, MatType());
      layerDeltas.resize(size, MatType());
      layerInputs.resize(size, MatType());

      layerGradients.resize(network.size(), MatType());

      outputDeltas.clear();
      inputDeltas.clear();
      accumulatedDeltas.clear();

      rowsCache.clear();
      slicesCache.clear();

      layerMemoryIsSet = false;
      validOutputDimensions = false;
      graphIsSet = false;
    }
  #endif
}

} // namespace mlpack

#endif
