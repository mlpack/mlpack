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

  virtual ~MultiLayer()
  {
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];
  }

  // TODO: implement these types of things...
//  MultiLayer(const MultiLayer& other);

  virtual void Forward(const InputType& input, OutputType& output)
  {
    // Make sure training/testing mode is set right in each layer.
    for (size_t i = 0; i < network.size(); ++i)
      network[i]->Training() = this->training;

    InitializeForwardPassMemory(input.n_cols);

    network.front()->Forward(input, layerOutputs.front());
    for (size_t i = 1; i < network.size(); ++i)
    {
      network[i]->Forward(layerOutputs[i - 1], layerOutputs[i]);
    }
  }

  virtual void Backward(const InputType& input,
                        const OutputType& gy,
                        OutputType& g)
  {
    InitializeBackwardPassMemory(input.n_cols);

    network.back()->Backward(layerOutputs.back(), gy, layerDeltas.back());

    for (size_t i = 1; i < network.size(); ++i)
    {
      network[network.size() - i]->Backward(
          layerOutputs[network.size() - 1],
          layerDeltas[network.size() - i + 1],
          layerDeltas[network.size() - i]);
    }
  }

  virtual void Gradient(const InputType& input,
                        const OutputType& error,
                        OutputType& gradient)
  {
    InitializeGradientPassMemory(gradient);

    // Pass gradients through each layer.
    // TODO: do we need to go back to front?  I guess not?
    network.front()->Gradient(input, layerDeltas[1], layerGradients.front());
    for (size_t i = 0; i < network.size() - 1; ++i)
    {
      network[i]->Gradient(layerOutputs[i - 1], layerDeltas[i + 1],
          layerGradients[i]);
    }

    network.back()->Gradient(layerOutputs[network.size() - 2], error,
        layerGradients[network.size() - 1]);
  }

  virtual void SetWeights(typename OutputType::elem_type* weightsPtr)
  {
    size_t start = 0;
    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->SetWeights(weightsPtr + start);
      start += network[i]->WeightSize();
    }
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
    inSize = std::accumulate(this->inputDimensions.begin(),
        this->inputDimensions.end(), 0);
    totalInputSize += inSize;

    for (size_t i = 1; i < network.size(); ++i)
    {
      network[i]->InputDimensions() = network[i - 1]->OutputDimensions();
      const size_t layerInputSize = std::accumulate(
          network[i]->InputDimensions().begin(),
          network[i]->InputDimensions().end(),
          0);

      totalInputSize += layerInputSize;
      totalOutputSize += layerInputSize;
    }

    const std::vector<size_t>& outputDimensions =
        network.back()->OutputDimensions();
    totalOutputSize += std::accumulate(outputDimensions.begin(),
        outputDimensions.end(), 0);
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
    ar(CEREAL_NVP(totalInputSize));
    ar(CEREAL_NVP(totalOutputSize));

    if (Archive::is_loading::value)
    {
      layerOutputMatrix.clear();
      layerDeltaMatrix.clear();
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
      const size_t layerInputSize = (i == 0) ?
          std::accumulate(this->inputDimensions.begin(),
              this->inputDimensions.end(), 0) :
          network[i - 1]->OutputSize();
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
      MakeAlias(layerGradients[i], gradient.colptr(gradientStart), weightSize,
          1);
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

#endif
