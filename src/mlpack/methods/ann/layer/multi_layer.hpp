/**
 * @file methods/ann/layer/multi_layer.hpp
 * @author Ryan Curtin
 *
 * Base class for neural network layers that are wrappers around other layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTI_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTI_LAYER_HPP

#include "layer.hpp"

namespace mlpack {

/**
 * A "multi-layer" is a layer that is a wrapper around other layers.  It passes
 * the input through all of its child layers sequentially, returning the output
 * from the last layer.
 *
 * It's likely not very useful to use this layer directly; instead, this layer
 * is meant as a base class for use by other layers that must store and use
 * multiple layers.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType>
class MultiLayer : public Layer<MatType>
{
 public:
  /**
   * Create an empty MultiLayer that holds no layers of its own.  Be sure to add
   * layers with Add() before using!
   */
  MultiLayer();

  //! Copy the given MultiLayer.
  MultiLayer(const MultiLayer& other);
  //! Take ownership of the layers of the given MultiLayer.
  MultiLayer(MultiLayer&& other);
  //! Copy the given MultiLayer.
  MultiLayer& operator=(const MultiLayer& other);
  //! Take ownership of the given MultiLayer.
  MultiLayer& operator=(MultiLayer&& other);

  //! Virtual destructor: delete all held layers.
  virtual ~MultiLayer()
  {
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];
  }

  //! Create a copy of the MultiLayer (this is safe for polymorphic use).
  virtual MultiLayer* Clone() const { return new MultiLayer(*this); }

  /**
   * Perform a forward pass with the given input data.  `output` is expected to
   * have the correct size (e.g. number of rows equal to `OutputSize()` of the
   * last held layer; number of columns equal to `input.n_cols`).
   *
   * @param input Input data to pass through the MultiLayer.
   * @param output Matrix to store output in.
   */
  virtual void Forward(const MatType& input, MatType& output);

  /**
   * Perform a forward pass with the given input data, but only on a subset of
   * the layers in the MultiLayer.  `output` is expected to have the correct
   * size (e.g. number of rows equal to `OutputSize()` of the last layer to be
   * computed; number of columns equal to `input.n_cols`).
   *
   * @param input Input data to pass through the MultiLayer.
   * @param output Matrix to store output in.
   * @param start Index of first layer to pass data through.
   * @param end Index of last layer to pass data through.
   */
  void Forward(const MatType& input,
               MatType& output,
               const size_t start,
               const size_t end);

  /**
   * Perform a backward pass with the given data.  `gy` is expected to be the
   * propagated error from the subsequent layer (or output), `input` is expected
   * to be the output from this layer when `Forward()` was called, and `g` will
   * store the propagated error from this layer (to be passed to the previous
   * layer as `gy`).
   *
   * It is expected that `g` has the correct size already (e.g., number of rows
   * equal to `OutputSize()` of the previous layer, and number of columns equal
   * to `input.n_cols`).
   *
   * This function is expected to be called for the same input data as
   * `Forward()` was just called for.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy Propagated error from next layer.
   * @param g Matrix to store propagated error in for previous layer.
   */
  virtual void Backward(const MatType& input,
                        const MatType& output,
                        const MatType& gy,
                        MatType& g);

  /**
   * Compute the gradients of each layer.
   *
   * This function is expected to be called for the same input data as
   * `Forward()` and `Backward()` were just called for.  That is, `input` here
   * should be the same data as `Forward()` was called with.
   *
   * `gradient` is expected to have the correct size already (e.g., number of
   * rows equal to 1, and number of columns equal to `WeightSize()`).
   *
   * @param input Original input data provided to Forward().
   * @param error Error as computed by `Backward()`.
   * @param gradient Matrix to store the gradients in.
   */
  virtual void Gradient(const MatType& input,
                        const MatType& error,
                        MatType& gradient);

  /**
   * Set the weights of the layer to use the memory given as `weightsPtr`.
   */
  virtual void SetWeights(const MatType& weightsIn);

  /**
   * Initialize the weight matrix of the layer.
   *
   * @param W Weight matrix to initialize.
   * @param elements Number of elements.
   */
  virtual void CustomInitialize(
      MatType& W,
      const size_t elements);

  /**
   * Return the number of weights in the MultiLayer.  This is the sum of the
   * number of weights in each layer.
   */
  virtual size_t WeightSize() const;

  /**
   * Compute the output dimensions of the MultiLayer using `InputDimensions()`.
   * This computes the dimensions of each layer held by the MultiLayer, and the
   * output dimensions are set to the output dimensions of the last layer.
   */
  virtual void ComputeOutputDimensions();

  /**
   * Compute the loss that should be added to the objective.
   */
  virtual double Loss() const;

  /**
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <typename LayerType, typename... Args>
  void Add(Args... args)
  {
    network.push_back(new LayerType(args...));
    layerOutputs.push_back(MatType());
    layerDeltas.push_back(MatType());
    layerGradients.push_back(MatType());
  }

  /**
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(Layer<MatType>* layer)
  {
    network.push_back(layer);
    layerOutputs.push_back(MatType());
    layerDeltas.push_back(MatType());
    layerGradients.push_back(MatType());
  }

  //! Get the network (series of layers) held by this MultiLayer.
  const std::vector<Layer<MatType>*>& Network() const
  {
    return network;
  }
  //! Modify the network (series of layers) held by this MultiLayer.  Be
  //! careful!
  std::vector<Layer<MatType>*>& Network() { return network; }

  //! Serialize the MultiLayer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 protected:
  /**
   * Initialize memory that will be used by each layer for the forward pass,
   * assuming that the input will have the given `batchSize`.  When `Forward()`
   * is called, each internally-held layer will output its results into the
   * memory allocated by this function (this is the internal member
   * `layerOutputMatrix` and its aliases `layerOutputs`).
   */
  void InitializeForwardPassMemory(const size_t batchSize);

  /**
   * Initialize memory that will be used by each layer for the backwards pass,
   * assuming that the input will have the given `batchSize`.  When `Backward()`
   * is called, each internally-held layer will output the results of its
   * backwards pass into the memory allocated by this function (this is the
   * internal member `layerDeltaMatrix` and its aliases `layerDeltas`).
   */
  void InitializeBackwardPassMemory(const size_t batchSize);

  /**
   * Initialize memory for the gradient pass.  This sets the internal aliases
   * `layerGradients` appropriately using the memory from the given `gradient`,
   * such that each layer will output its gradient (via its `Gradient()` method)
   * into the appropriate member of `layerGradients`.
   */
  void InitializeGradientPassMemory(MatType& gradient);

  //! The internally-held network.
  std::vector<Layer<MatType>*> network;

  // Total number of elements in the input, cached for convenience.
  size_t inSize;
  // Total number of input elements for *every* layer.
  size_t totalInputSize;
  // Total number of output elements for *every* layer.
  size_t totalOutputSize;

  //! This matrix stores all of the outputs of each layer when Forward() is
  //! called.  See `InitializeForwardPassMemory()`.
  MatType layerOutputMatrix;
  //! These are aliases of `layerOutputMatrix` for each layer.
  std::vector<MatType> layerOutputs;

  //! This matrix stores all of the backwards pass results of each layer when
  //! Backward() is called.  See `InitializeBackwardPassMemory()`.
  MatType layerDeltaMatrix;
  //! These are aliases of `layerDeltaMatrix` for each layer.
  std::vector<MatType> layerDeltas;

  //! Gradient aliases for each layer.  Note that this is *only* valid in the
  //! context of `Gradient()`!  We have it as a class member to avoid
  //! reallocating the `MatType`s each call to `Gradient()`.
  std::vector<MatType> layerGradients;
};

} // namespace mlpack

// Include implementation.
#include "multi_layer_impl.hpp"

#endif
