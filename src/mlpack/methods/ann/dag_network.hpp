/**
 * @file methods/ann/dag_network.hpp
 * @author Andrew Furey
 *
 * Definition of the DAGNetwork class, which allows uers to describe a
 * computational graph to build arbitrary neural networks with skip
 * connections.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_DAG_NETWORK_HPP
#define MLPACK_METHODS_ANN_DAG_NETWORK_HPP

#include <mlpack/core.hpp>

#include "init_rules/init_rules.hpp"

#include <ensmallen.hpp>

namespace mlpack {

/**
 * Implementation of a direct acyclic graph. Any layer that inherits
 * from the base `Layer` class can be added to this model.
 *
 * A network can be created by using the `Add()` method to add
 * layers to the network. Each layer is then linked using `Connect()`.
 * A node with multiple parents will concatenate the output of its
 * parents along a specified axis. You can specify the axis of
 * concatenation with `SetAxis`. If the axis is not specifed, the default
 * axis will be used, which is 0.
 *
 * A DAGNetwork cannot have any cycles. Creating a network with a cycle will
 * result in an error.
 *
 * A DAGNetwork can only have one input layer and one output layer.
 *
 * Although the actual types passed as input will be matrix objects with one
 * data point per column, each data point can be a tensor of arbitrary shape.
 * If data points are not 1-dimensional vectors, then set the shape of the input
 * with `InputDimensions()` before calling `Train()`.
 *
 * More granular functionality is available with `Forward()`, Backward()`, and
 * `Evaluate()`, or even by accessing the individual layers directly with
 * `Network()`.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam MatType Type of matrix to be given as input to the network.
 */
template<
    typename OutputLayerType = NegativeLogLikelihood,
    typename InitializationRuleType = RandomInitialization,
    typename MatType = arma::mat>
class DAGNetwork
{
 public:
  /**
   * Create the DAGNetwork object.
   *
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param outputLayer Output layer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  DAGNetwork(OutputLayerType outputLayer = OutputLayerType(),
             InitializationRuleType initializeRule = InitializationRuleType());

  // Copy constructor.
  DAGNetwork(const DAGNetwork& other);
  // Move constructor.
  DAGNetwork(DAGNetwork&& other);
  // Copy operator.
  DAGNetwork& operator=(const DAGNetwork& other);
  // Move assignment operator.
  DAGNetwork& operator=(DAGNetwork&& other);

  // Destructor: delete all layers.
  ~DAGNetwork()
  {
    for (size_t i = 0; i < network.size(); i++)
      delete network[i];
  }


  using CubeType = typename GetCubeType<MatType>::type;

  /**
   * Add a new layer to the model.  Note that any trainable weights of this
   * layer will be reset!  (Any constant parameters are kept.) This layer
   * should only receive input from one layer.
   *
   * @param layer The Layer to be added to the model.
   *
   * returns the index of the layer in `network`, to be used in `Connect()`
   */
  template <typename LayerType, typename... Args>
  size_t Add(Args&&... args)
  {
    size_t id = network.size();
    network.push_back(new LayerType(std::forward<Args>(args)...));
    AddLayer(id);

    return id;
  }

  template <template<typename...> typename LayerType, typename... Args>
  size_t Add(Args&&... args)
  {
    size_t id = network.size();
    network.push_back(new LayerType<MatType>(std::forward<Args>(args)...));
    AddLayer(id);

    return id;
  }

  template <typename LayerType>
  size_t Add(LayerType&& layer,
             typename std::enable_if_t<
                 !std::is_pointer_v<std::remove_reference_t<LayerType>>>* = 0)
  {
    using NewLayerType =
        typename std::remove_cv_t<std::remove_reference_t<LayerType>>;

    size_t id = network.size();
    network.push_back(new NewLayerType(std::forward<LayerType>(layer)));
    AddLayer(id);

    return id;
  }

  /**
   * Set the axis to concatenate along for a layer that expects multiple
   * parent node. Can only be set once per layer.
   *
   * @param concatAxis The axis to concatenate parent node outputs along.
   * @param layerId The layer to be added to the model.
   */
  void SetAxis(size_t layerId, size_t concatAxis);

  /**
   * Create an edge between two layers. If the child node expects multiple
   * parents, the child must have been added to the network with an axis.
   *
   * @param inputLayer The parent node whose output is the input to `outputLayer`
   * @param outputLayer The child node whose input will come from `inputLayer`
   */
  void Connect(size_t parentNodeId, size_t childNodeId);

  // Get the layers of the network, in the order the user specified.
  const std::vector<Layer<MatType>*>& Network() const
  {
    return network;
  }

  // Get the layers of the network, in topological order.
  const std::vector<Layer<MatType>*> SortedNetwork()
  {
    if (!graphIsSet)
      CheckGraph();

    std::vector<Layer<MatType>*> sortedLayers;
    for (size_t i = 0; i < sortedNetwork.size(); i++)
    {
      size_t layerIndex = sortedNetwork[i];
      sortedLayers.push_back(network[layerIndex]);
    }
    return sortedLayers;
  }



  template<typename OptimizerType, typename... CallbackTypes>
  typename MatType::elem_type Train(MatType predictors,
                                    MatType responses,
                                    OptimizerType& optimizer,
                                    CallbackTypes&&... callbacks);

  template<typename OptimizerType = ens::RMSProp, typename... CallbackTypes>
  typename MatType::elem_type Train(MatType predictors,
                                    MatType responses,
                                    CallbackTypes&&... callbacks);

  /**
   * Predict the responses to a given set of predictors. The responses will be
   * the output of the output layer when `predictors` is passed through the
   * whole network (`OutputLayerType`).
   *
   * @param predictors Input predictors.
   * @param results Matrix to put output predictions of responses into.
   * @param batchSize Batch size to use for prediction.
   */
  void Predict(const MatType& predictors,
               MatType& results,
               const size_t batchSize = 128);

  // Return the number of weights in the model.
  size_t WeightSize();

  /**
   * Set the logical dimensions of the input.  `Train()` and `Predict()` expect
   * data to be passed such that one point corresponds to one column, but this
   * data is allowed to be an arbitrary higher-order tensor.
   *
   * So, if the input is meant to be 28x28x3 images, then the
   * input data to `Train()` or `Predict()` should have 28*28*3 = 2352 rows, and
   * `InputDimensions()` should be set to `{ 28, 28, 3 }`.  Then, the layers of
   * the network will interpret each input point as a 3-dimensional image
   * instead of a 1-dimensional vector.
   *
   * If `InputDimensions()` is left unset before training, the data will be
   * assumed to be a 1-dimensional vector.
   */
  std::vector<size_t>& InputDimensions()
  {
    validOutputDimensions = false;
    graphIsSet = false;
    layerMemoryIsSet = false;

    return inputDimensions;
  }

  // Get the logical dimensions of the input.
  const std::vector<size_t>& InputDimensions() const { return inputDimensions; }

  const std::vector<size_t>& OutputDimensions()
  {
    if (!graphIsSet)
      CheckGraph();

    if (!validOutputDimensions)
      UpdateDimensions("DAGNetwork::OutputDimensions()");

    size_t lastLayer = sortedNetwork.back();
    return network[lastLayer]->OutputDimensions();
  }

  // Return the current set of weights.  These are linearized: this contains
  // the weights of every layer.
  const MatType& Parameters() const { return parameters; }
  // Modify the current set of weights.  These are linearized: this contains
  // the weights of every layer.  Be careful!  If you change the shape of
  // `parameters` to something incorrect, it may be re-initialized the next
  // time a forward pass is done.
  MatType& Parameters() { return parameters; }

  /**
   * Reset the stored data of the network entirely.  This resets all weights of
   * each layer using `InitializationRuleType`, and prepares the network to
   * accept a (flat 1-d) input size of `inputDimensionality` (if passed), or
   * whatever input size has been set with `InputDimensions()`.
   *
   * If no input size has been set with `InputDimensions()`, and
   * `inputDimensionality` is 0, an exception will be thrown, since an empty
   * input size is invalid.
   *
   * This also resets the mode of the network to prediction mode (not training
   * mode).  See `SetNetworkMode()` for more information.
   */
  void Reset(const size_t inputDimensionality = 0);

  /**
   * Set all the layers in the network to training mode, if `training` is
   * `true`, or set all the layers in the network to testing mode, if `training`
   * is `false`.
   */
  void SetNetworkMode(const bool training);

  /**
   * Perform a manual forward pass of the data.
   *
   * `Forward()` and `Backward()` should be used as a pair, and they are
   * designed mainly for advanced users. You should try to use `Predict()` and
   * `Train()`, if you can.
   *
   * @param inputs The input data.
   * @param results The predicted results.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Perform a manual backward pass of the data.
   *
   * `Forward()` and `Backward()` should be used as a pair, and they are
   * designed mainly for advanced users. You should try to use `Predict()` and
   * `Train()` instead, if you can.
   *
   * @param input Input of the network
   * @param output Output of the network
   * @param error  Error from loss function.
   * @param gradients Computed gradients.
   */
  void Backward(const MatType& input,
                const MatType& output,
                const MatType& error,
                MatType& gradients);

  /**
   * Evaluate the network with the given predictors and responses.
   * This functions is usually used to monitor progress while training.
   *
   * @param predictors Input variables.
   * @param responses Target outputs for input variables.
   */
  typename MatType::elem_type Evaluate(const MatType& predictors,
                                       const MatType& responses);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //
  // Only ensmallen utility functions for training are found below here.
  // They aren't generally useful otherwise.
  //

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the network with the given parameters.
   *
   * @param parameters Matrix model parameters.
   */
  typename MatType::elem_type Evaluate(const MatType& parameters);

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the network with the given parameters, but using only
   * a number of data points. This is useful for optimizers such as SGD, which
   * require a separable objective function.
   *
   * Note that the network may return different results depending on the mode it
   * is in (see `SetNetworkMode()`).
   *
   * @param parameters Matrix model parameters.
   * @param begin Index of the starting point to use for objective function
   *        evaluation.
   * @param batchSize Number of points to be passed at a time to use for
   *        objective function evaluation.
   */
  typename MatType::elem_type Evaluate(const MatType& parameters,
                                       const size_t begin,
                                       const size_t batchSize);

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the network with the given parameters.
   * This function is usually called by the optimizer to train the model.
   * This just calls the overload of EvaluateWithGradient() with batchSize = 1.
   *
   * @param parameters Matrix model parameters.
   * @param gradient Matrix to output gradient into.
   */
  typename MatType::elem_type EvaluateWithGradient(const MatType& parameters,
                                                   MatType& gradient);


  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the network with the given parameters, but using only
   * a number of data points. This is useful for optimizers such as SGD, which
   * require a separable objective function.
   *
   * @param parameters Matrix model parameters.
   * @param begin Index of the starting point to use for objective function
   *        evaluation.
   * @param gradient Matrix to output gradient into.
   * @param batchSize Number of points to be passed at a time to use for
   *        objective function evaluation.
   */

  typename MatType::elem_type EvaluateWithGradient(const MatType& parameters,
                                                   const size_t begin,
                                                   MatType& gradient,
                                                   const size_t batchSize);

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Evaluate the gradient of the network with the given parameters,
   * and with respect to only a number of points in the dataset. This is useful
   * for optimizers such as SGD, which require a separable objective function.
   *
   * @param parameters Matrix of the model parameters to be optimized.
   * @param begin Index of the starting point to use for objective function
   *        gradient evaluation.
   * @param gradient Matrix to output gradient into.
   * @param batchSize Number of points to be processed as a batch for objective
   *        function gradient evaluation.
   */
  void Gradient(const MatType& parameters,
                const size_t begin,
                MatType& gradient,
                const size_t batchSize);


  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Return the number of separable functions (the number of predictor points).
   */
  size_t NumFunctions() const { return responses.n_cols; }

  /**
   * Note: this function is implemented so that it can be used by ensmallen's
   * optimizers.  It's not generally meant to be used otherwise.
   *
   * Shuffle the order of function visitation.  (This is equivalent to shuffling
   * the dataset during training.)
   */
  void Shuffle();

  /**
   * Prepare the network for training on the given data.
   *
   * This function won't actually trigger the training process, and is
   * generally only useful internally.
   *
   * @param predictors Input data variables.
   * @param responses Outputs results from input data variables.
   */
  void ResetData(MatType predictors, MatType responses);

 private:
  // Helper functions.

  void AddLayer(size_t nodeId)
  {
    layerGradients.push_back(MatType());
    childrenList.insert({ nodeId, {} });
    parentsList.insert({ nodeId, {} });

    if (network.size() > 1)
    {
      layerOutputs.push_back(MatType());
      layerInputs.push_back(MatType());
      layerDeltas.push_back(MatType());
    }

    validOutputDimensions = false;
    graphIsSet = false;
    layerMemoryIsSet = false;
  }

  // Use the InitializationPolicy to initialize all the weights in the network.
  void InitializeWeights();

  // Call each layers `CustomInitialize`
  void CustomInitialize(MatType& W, const size_t elements);

  // Make the memory of each layer point to the right place, by calling
  // SetWeights() on each layer.
  void SetLayerMemory();

  /**
   * Ensure that all the there are no cycles in the graph and that the graph
   * has one input and one output only. It will topologically sort the network
   * for the forward and backward passes. This will set `graphIsSet` to true
   * only if the graph is valid and topologically sorted.
   */
  void CheckGraph();

  /**
   * Ensure that all the locally-cached information about the network is valid,
   * all parameter memory is initialized, and we can make forward and backward
   * passes.
   *
   * @param functionName Name of function to use if an exception is thrown.
   * @param inputDimensionality Given dimensionality of the input data.
   * @param setMode If true, the mode of the network will be set to the
   *     parameter given in `training`.  Otherwise the mode of the network is
   *     left unmodified.
   * @param training Mode to set the network to; `true` indicates the network
   *     should be set to training mode; `false` indicates testing mode.
   */
  void CheckNetwork(const std::string& functionName,
                    const size_t inputDimensionality,
                    const bool setMode = false,
                    const bool training = false);

  /**
   * This computes the dimensions of each layer held by the network, and the
   * output dimensions are set to the output dimensions of the last layer.
   *
   * Layers with multiple inputs need an axis to concatenate their output
   * along, specifed in Add(). Every dimension not along that axis in each
   * input tensor must be the same, while the dimension along that axis can
   * vary.
   */
  void ComputeOutputDimensions();

  /**
   * Set the input and output dimensions of each layer in the network correctly.
   * The size of the input is taken, in case `inputDimensions` has not been set
   * otherwise (e.g. via `InputDimensions()`).  If `InputDimensions()` is not
   * empty, then `inputDimensionality` is ignored.
   */
  void UpdateDimensions(const std::string& functionName,
                        const size_t inputDimensionality = 0);

  /**
   * Set the weights of the layers
   */
  void SetWeights(const MatType& weightsIn);

  /**
   * Initialize memory that will be used by each layer for the forward pass,
   * assuming that the input will have the given `batchSize`.  When `Forward()`
   * is called, `layerOutputMatrix` is allocated with enough memory to fit
   * the outputs of each layer and to hold concatenations of output layers
   * as inputs into layers as specified by `Add()` and `Connect()`.
   */
  void InitializeForwardPassMemory(const size_t batchSize);

  /**
   * TODO: explain how the backward pass memory works.
   */
  void InitializeBackwardPassMemory(const size_t batchSize);

  /**
   * Initialize memory for the gradient pass.  This sets the internal aliases
   * `layerGradients` appropriately using the memory from the given `gradient`,
   * such that each layer will output its gradient (via its `Gradient()` method)
   * into the appropriate member of `layerGradients`.
   */
  void InitializeGradientPassMemory(MatType& gradient);

  /**
   * Compute the loss that should be added to the objective for each layer.
   */
  double Loss() const;

  /**
   * Check if the optimizer has MaxIterations() parameter, if it does then check
   * if its value is less than the number of datapoints in the dataset.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param optimizer optimizer used in the training process.
   * @param samples Number of datapoints in the dataset.
   */
  template<typename OptimizerType>
  std::enable_if_t<
      ens::traits::HasMaxIterationsSignature<OptimizerType>::value, void>
  WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const;

  /**
   * Check if the optimizer has MaxIterations() parameter; if it doesn't then
   * simply return from the function.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @param optimizer optimizer used in the training process.
   * @param samples Number of datapoints in the dataset.
   */
  template<typename OptimizerType>
  std::enable_if_t<
      !ens::traits::HasMaxIterationsSignature<OptimizerType>::value, void>
  WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const;

  // Instantiated output layer used to evaluate the network.
  OutputLayerType outputLayer;

  // Instantiated InitializationRule object for initializing the network
  // parameter.
  InitializationRuleType initializeRule;

  // The internally-held network, sorted in the order that the user
  // specified when using `Add()`
  std::vector<Layer<MatType>*> network;

  // The internally-held network, sorted topologically when `CheckNetwork`
  // is called if the graph is valid.
  std::vector<size_t> sortedNetwork;

  // The internally-held map of nodes that holds its edges to outgoing nodes.
  // Uses network indices as keys.
  std::unordered_map<size_t, std::vector<size_t>> childrenList;

  // The internally-held map of nodes that holds its edges to incoming nodes.
  // Uses network indices as keys.
  std::unordered_map<size_t, std::vector<size_t>> parentsList;

  // The internally-held map of what axes to concatenate along for each layer
  // with multiple inputs
  // Uses network indices as keys.
  std::unordered_map<size_t, size_t> layerAxes;

  // Map layer index in network to layer index in sortedNetwork
  // Uses network indices as keys.
  std::unordered_map<size_t, size_t> sortedIndices;

  /**
   * Matrix of (trainable) parameters.  Each weight here corresponds to a layer,
   * and each layer's `parameters` member is an alias pointing to parameters in
   * this matrix.
   *
   * Note: although each layer may have its own MatType and MatType,
   * ensmallen optimization requires everything to be stored in one matrix
   * object, so we have chosen MatType.  This could be made more flexible
   * with a "wrapper" class implementing the Armadillo API.
   */
  MatType parameters;

  // Dimensions of input data.
  std::vector<size_t> inputDimensions;

  //! The matrix of data points (predictors).  This member is empty, except
  //! during training---we must store a local copy of the training data since
  //! the ensmallen optimizer will not provide training data.
  MatType predictors;

  //! The matrix of responses to the input data points.  This member is empty,
  //! except during training.
  MatType responses;

  // Locally-stored output of the network from a forward pass; used by the
  // backward pass.
  MatType networkOutput;
  //! Locally-stored output of the backward pass; used by the gradient pass.
  MatType error;

  // This matrix stores all of the outputs of each layer when `Forward()` is
  // called.  See `InitializeForwardPassMemory()`.
  MatType layerOutputMatrix;
  // These are aliases of `layerOutputMatrix` for the input of each layer
  std::vector<MatType> layerInputs;
  // These are aliases of `layerOutputMatrix` for the output of each layer.
  std::vector<MatType> layerOutputs;

  // Memory for the backward pass.
  MatType layerDeltaMatrix;

  // Needed in case the first layer is a `MultiLayer` so that its
  // gradients are calculated.
  MatType networkDelta;

  // A layers delta Loss w.r.t delta Outputs.
  std::vector<MatType> layerDeltas;

  // A layers output deltas. Uses sortedNetwork indices as keys.
  std::unordered_map<size_t, MatType> outputDeltas;

  // A layers input deltas. Uses sortedNetwork indices as keys.
  std::unordered_map<size_t, MatType> inputDeltas;

  // A layers accumulated deltas, for layers with multiple children.
  // Uses sortedNetwork indices as keys.
  std::unordered_map<size_t, MatType> accumulatedDeltas;

  // Gradient aliases for each layer.
  std::vector<MatType> layerGradients;

  // Cache of rows for concatenation.
  std::unordered_map<size_t, size_t> rowsCache;
  // Cache of slices for concatenation.
  std::unordered_map<size_t, size_t> slicesCache;

  // If true, each layer has its inputDimensions properly set.
  bool validOutputDimensions;

  // If true, the graph is valid and has been topologically sorted.
  bool graphIsSet;

  // If true, each layer has its activation/gradient memory properly set
  // for the forward/backward pass.
  bool layerMemoryIsSet;

  bool extraDeltasAllocated;
};

} // namespace mlpack

#include "dag_network_impl.hpp"

#endif
