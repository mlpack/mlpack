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

#include "../make_alias.hpp"

namespace mlpack {
namespace ann {

/**
 * A "multi-layer" is a layer that is a wrapper around other layers.
 *
 * TODO: better comments
 */
template<typename InputType, typename OutputType>
class MultiLayer : public Layer<InputType, OutputType>
{
 public:
  // TODO: implement these types of things...
  MultiLayer();
  MultiLayer(const MultiLayer& other);
  MultiLayer(MultiLayer&& other);
  MultiLayer& operator=(const MultiLayer& other);
  MultiLayer& operator=(MultiLayer&& other);

  virtual ~MultiLayer()
  {
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];
  }


  virtual MultiLayer* Clone() const { return new MultiLayer(*this); }

  /**
   * Perform a forward pass with the given input data.  `output` is expected to
   * have the correct size (e.g. number of rows equal to `OutputSize()` of the
   * last held layer; number of columns equal to `input.n_cols`).
   *
   * @param input Input data to pass through the MultiLayer.
   * @param output Matrix to store output in.
   */
  virtual void Forward(const InputType& input, OutputType& output);

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
  void Forward(const InputType& input,
               OutputType& output,
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
   * @param input Output of Forward().
   * @param gy Propagated error from next layer.
   * @param g Matrix to store propagated error in for previous layer.
   */
  virtual void Backward(const InputType& input,
                        const OutputType& gy,
                        OutputType& g);

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
  virtual void Gradient(const InputType& input,
                        const OutputType& error,
                        OutputType& gradient);

  virtual void SetWeights(typename OutputType::elem_type* weightsPtr)
  {
    size_t start = 0;
    const size_t totalWeightSize = WeightSize();
    for (size_t i = 0; i < network.size(); ++i)
    {
      const size_t weightSize = network[i]->WeightSize();

      // Sanity check: ensure we aren't passing memory past the end of the
      // parameters.
      Log::Assert(start + weightSize <= totalWeightSize,
          "FNN::SetLayerMemory(): parameter size does not match total layer "
          "weight size!");

      network[i]->SetWeights(weightsPtr + start);
      start += weightSize;
    }

    // Technically this check should be unnecessary, but there's nothing wrong
    // with a little paranoia...
    Log::Assert(start == totalWeightSize,
        "FNN::SetLayerMemory(): total layer weight size does not match parameter "
        "size!");
  }

  virtual size_t OutputSize() const
  {
    // Return the output size of the last layer.
    return network.back()->OutputSize();
  }

  virtual size_t WeightSize() const
  {
    // Sum the weights in each layer.
    size_t total = 0;
    for (size_t i = 0; i < network.size(); ++i)
      total += network[i]->WeightSize();
    return total;
  }

  virtual void ComputeOutputDimensions()
  {
    inSize = 0;
    totalInputSize = 0;
    totalOutputSize = 0;

    // Propagate the input dimensions forward to the output.
    network.front()->InputDimensions() = this->inputDimensions;
    inSize = this->inputDimensions[1];
    for (size_t i = 1; i < this->inputDimensions.size(); ++i)
      inSize *= this->inputDimensions[i];
    totalInputSize += inSize;

    for (size_t i = 1; i < network.size(); ++i)
    {
      network[i]->InputDimensions() = network[i - 1]->OutputDimensions();
      size_t layerInputSize = network[i]->InputDimensions()[0];
      for (size_t j = 1; j < network[i]->InputDimensions().size(); ++j)
        layerInputSize *= network[i]->InputDimensions()[j];

      totalInputSize += layerInputSize;
      totalOutputSize += layerInputSize;
    }

    size_t lastLayerSize = network.back()->OutputDimensions()[0];
    for (size_t i = 1; i < network.back()->OutputDimensions().size(); ++i)
      lastLayerSize *= network.back()->OutputDimensions()[i];

    totalOutputSize += lastLayerSize;
    this->outputDimensions = network.back()->OutputDimensions();
  }

  virtual double Loss() const
  {
    double loss = 0.0;
    for (size_t i = 0; i < network.size(); ++i)
      loss += network[i]->Loss();

    return loss;
  }

  /*
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args)
  {
    network.push_back(new LayerType(args...));
    layerOutputs.push_back(OutputType());
    layerDeltas.push_back(OutputType());
    layerGradients.push_back(OutputType());
  }

  // TODO: handle network ownership?

  /*
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(Layer<InputType, OutputType>* layer)
  {
    network.push_back(layer);
    layerOutputs.push_back(OutputType());
    layerDeltas.push_back(OutputType());
    layerGradients.push_back(OutputType());
  }

  const std::vector<Layer<InputType, OutputType>*> Network() const { return
network; }
  std::vector<Layer<InputType, OutputType>*>& Network() { return network; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(cereal::base_class<Layer<InputType, OutputType>>(this));

    ar(CEREAL_VECTOR_POINTER(network));
    ar(CEREAL_NVP(inSize));
    ar(CEREAL_NVP(totalInputSize));
    ar(CEREAL_NVP(totalOutputSize));

    if (Archive::is_loading::value)
    {
      layerOutputMatrix.clear();
      layerDeltaMatrix.clear();
      layerGradients.clear();
      layerOutputs.resize(network.size(), OutputType());
      layerDeltas.resize(network.size(), OutputType());
      layerGradients.resize(network.size(), OutputType());
    }
  }

 protected:

  void InitializeForwardPassMemory(const size_t batchSize)
  {
    // We need to initialize memory to store the output of each layer's
    // Forward() call.  We'll do this all in one matrix, but, the size of this
    // matrix depends on the batch size we are using for computation.  We avoid
    // resizing layerOutputMatrix down, unless we only need 10% or less of it.
    if (batchSize * totalOutputSize > layerOutputMatrix.n_elem ||
        batchSize * totalOutputSize <
            std::floor(0.1 * layerOutputMatrix.n_elem))
    {
      // All outputs will be represented by one big block of memory.
      layerOutputMatrix = OutputType(1, batchSize * totalOutputSize);
    }

    // Now, create an alias to the right place for each layer.  We assume that
    // layerOutputs is already sized correctly (this should be done by Add()).
    size_t start = 0;
    for (size_t i = 0; i < layerOutputs.size(); ++i)
    {
      const size_t layerOutputSize = network[i]->OutputSize();
      MakeAlias(layerOutputs[i], layerOutputMatrix.colptr(start),
          layerOutputSize, batchSize);
      start += batchSize * layerOutputSize;
    }
  }

  void InitializeBackwardPassMemory(const size_t batchSize)
  {
    // We need to initialize memory to store the output of each layer's
    // Backward() call.  We do this similarly to InitializeForwardPassMemory(),
    // but we must store a matrix to use as the delta for each layer.
    if (batchSize * totalInputSize > layerDeltaMatrix.n_elem ||
        batchSize * totalInputSize < std::floor(0.1 * layerOutputMatrix.n_elem))
    {
      // All deltas will be represented by one big block of memory.
      layerDeltaMatrix = OutputType(1, batchSize * totalInputSize);
    }

    // Now, create an alias to the right place for each layer.  We assume that
    // layerDeltas is already sized correctly (this should be done by Add()).
    size_t start = 0;
    for (size_t i = 0; i < layerDeltas.size(); ++i)
    {
      size_t layerInputSize = 1;
      if (i == 0)
      {
        for (size_t j = 0; j < this->inputDimensions.size(); ++j)
          layerInputSize *= this->inputDimensions[j];
      }
      else
      {
        layerInputSize = network[i - 1]->OutputSize();
      }
      MakeAlias(layerDeltas[i], layerDeltaMatrix.colptr(start), layerInputSize,
          batchSize);
      start += batchSize * layerInputSize;
    }
  }

  void InitializeGradientPassMemory(OutputType& gradient)
  {
    // We need to initialize memory to store the gradients of each layer.
    // To do this, we need to know the weight size of each layer.
    size_t gradientStart = 0;
    for (size_t i = 0; i < network.size(); ++i)
    {
      const size_t weightSize = network[i]->WeightSize();
      MakeAlias(layerGradients[i], gradient.memptr() + gradientStart,
          weightSize, 1);
      gradientStart += weightSize;
    }
  }

  std::vector<Layer<InputType, OutputType>*> network;

  // Total number of elements in the input, cached for convenience.
  size_t inSize;
  // Total number of input elements for *every* layer.
  size_t totalInputSize;
  // Total number of output elements for *every* layer.
  size_t totalOutputSize;

  OutputType layerOutputMatrix;
  std::vector<OutputType> layerOutputs;
  OutputType layerDeltaMatrix;
  std::vector<OutputType> layerDeltas;
  std::vector<OutputType> layerGradients;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "multi_layer_impl.hpp"

#endif
