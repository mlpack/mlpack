/**
 * @file inception_layer.hpp
 * @author Nilay Jain
 *
 * Definition of the InceptionLayer class.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/pooling_layer.hpp>
#include <mlpack/methods/ann/pooling_rules/max_pooling.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/network_util.hpp>
namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


//! input --> inception_layer --> output.

/*

Inception module of GoogLeNet.

It applies four different functions to the input array and concatenates
their outputs along the channel dimension. Three of them are 2D
convolutions of sizes 1x1, 3x3 and 5x5. Convolution paths of 3x3 and 5x5
sizes have 1x1 convolutions (called projections) ahead of them. The other
path consists of 1x1 convolution (projection) and 3x3 max pooling.

The output array has the same spatial size as the input. In order to
satisfy this, Inception module uses appropriate padding for each
convolution and pooling.

See: `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.

*/

template<typename InputDataType = arma::cube,
         typename OutputDataType = arma::cube>
class InceptionLayer
{
 public:
  //! Locally-stored number of input maps.
  size_t inMaps;

  //! bias value
  size_t bias;

  //! Locally-stored outMaps of the constituent layers of inception layer.
  size_t out1, out3, out5, projSize3, projSize5, poolProj;

  //! Locally-stored convLayers.
  ConvLayer<> conv1, proj3, conv3, proj5, conv5, convPool;

  //! Locally-stored baseLayers.
  BaseLayer2D<RectifierFunction> base1, baseProj3, base3, baseProj5, base5, basePool;
  
  //! Locally-stored biasLayers.
  BiasLayer2D<> bias1, biasProj5, biasProj3, bias3, bias5, biasPool;
  
  //! Locally-stored poolLayer.
  PoolingLayer<MaxPooling> pool3;
  /**
   *
   * @param inMaps The number of input maps.
   * @param outMaps The number of output maps.
   *
    */
  InceptionLayer( const size_t inMaps,
                  const size_t out1,
                  const size_t projSize3,
                  const size_t out3,
                  const size_t projSize5,
                  const size_t out5,
                  const size_t poolProj,
                  const size_t bias = 0) :
      inMaps(inMaps),
      bias(bias),
      out1(out1),
      out3(out3),
      out5(out5),
      projSize5(projSize5),
      projSize3(projSize3),
      poolProj(poolProj),
      conv1(inMaps, out1, 1, 1),
      proj3(inMaps, projSize3, 1, 1),
      conv3(projSize3, out3, 3, 3, 1, 1, 1, 1),
      proj5(inMaps, projSize5, 1, 1),
      conv5(projSize5, out5, 5, 5, 1, 1, 2, 2),
      convPool(inMaps, poolProj, 1, 1, 1, 1, 1, 1),
      pool3(3),
      bias1(out1, bias),
      biasProj3(projSize3, bias),
      bias3(out3, bias),
      biasProj5(projSize5, bias),
      bias5(out5, bias),
      biasPool(poolProj, bias)
  {
    /* Nothing to do. */
  }

  // perform forward passes for all the layers.

  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    conv1.InputParameter() = input;

    // Forward pass for 1x1 conv path.
    conv1.Forward(conv1.InputParameter(), conv1.OutputParameter());
    bias1.Forward(conv1.OutputParameter(), bias1.OutputParameter());
    base1.InputParameter() = bias1.OutputParameter();
    base1.Forward(bias1.OutputParameter(), base1.OutputParameter());

    // Forward pass for 3x3 conv path.
    proj3.InputParameter() = input;
    proj3.Forward(input, proj3.OutputParameter());
    biasProj3.Forward(proj3.OutputParameter(), biasProj3.OutputParameter());
    baseProj3.InputParameter() = biasProj3.OutputParameter();
    baseProj3.Forward(biasProj3.OutputParameter(), baseProj3.OutputParameter());
    conv3.InputParameter() = baseProj3.OutputParameter();
    conv3.Forward(baseProj3.OutputParameter(), conv3.OutputParameter());
    bias3.Forward(conv3.OutputParameter(), bias3.OutputParameter());
    base3.InputParameter() = bias3.OutputParameter();
    base3.Forward(bias3.OutputParameter(), base3.OutputParameter());    
    
    // Forward pass for 5x5 conv path.
    proj5.InputParameter() = input;
    proj5.Forward(input, proj5.OutputParameter());
    biasProj5.Forward(proj5.OutputParameter(), biasProj5.OutputParameter());
    baseProj5.InputParameter() = biasProj5.OutputParameter();
    baseProj5.Forward(biasProj5.OutputParameter(), baseProj5.OutputParameter());
    conv5.InputParameter() = baseProj5.OutputParameter();
    conv5.Forward(baseProj5.OutputParameter(), conv5.OutputParameter());
    bias5.Forward(conv5.OutputParameter(), bias5.OutputParameter());
    base5.InputParameter() = bias5.OutputParameter();
    base5.Forward(bias5.OutputParameter(), base5.OutputParameter());
    
    // Forward pass for 3x3 pool path. 
    pool3.InputParameter() = input;
    pool3.Forward(input, pool3.OutputParameter());
    Pad(pool3.OutputParameter(), 1, 1, convPool.InputParameter());
    convPool.Forward(pool3.OutputParameter(), convPool.OutputParameter());
    biasPool.Forward(convPool.OutputParameter(), biasPool.OutputParameter());
    basePool.InputParameter() = biasPool.OutputParameter();
    basePool.Forward(convPool.OutputParameter(), basePool.OutputParameter());

    // concatenate outputs of all the paths.
    output = arma::join_slices( 
              arma::join_slices(
                arma::join_slices( 
                  base1.OutputParameter(), base3.OutputParameter() ), 
                  base5.OutputParameter() ), basePool.OutputParameter());

  }

  //! perform backward passes for all the layers.
  
  // Backward(error, network)
  // error : backpropagated error
  // g : calcualted gradient.
  // populate delta for all the layers.
  // size of delta = size of inputParameter.
  template<typename eT>
  void Backward(arma::Cube<eT>&, arma::Cube<eT>& error, arma::Cube<eT>& )
  {
    InputDataType in;
    size_t slice_idx = 0;

    arma::cube subError = error.slices(slice_idx, slice_idx + base1.OutputParameter().n_slices - 1);
    slice_idx += base1.OutputParameter().n_slices;
    base1.Backward(base1.OutputParameter(), subError, base1.Delta());
    bias1.Backward(bias1.OutputParameter(), base1.Delta(), bias1.Delta());
    conv1.Backward(conv1.OutputParameter(), bias1.Delta(), conv1.Delta());

    subError = error.slices(slice_idx, slice_idx + base3.OutputParameter().n_slices - 1);
    slice_idx += base3.OutputParameter().n_slices;
    base3.Backward(base3.OutputParameter(), subError, base3.Delta());
    bias3.Backward(bias3.OutputParameter(), base3.Delta(), bias3.Delta());
    conv3.Backward(conv3.OutputParameter(), bias3.Delta(), conv3.Delta());
    baseProj3.Backward(baseProj3.OutputParameter(), conv3.Delta(), baseProj3.Delta());
    biasProj3.Backward(biasProj3.OutputParameter(), baseProj3.Delta(), biasProj3.Delta());
    proj3.Backward(proj3.OutputParameter(), biasProj3.Delta(), proj3.Delta());

    subError = error.slices(slice_idx, slice_idx + base5.OutputParameter().n_slices - 1);
    slice_idx += base5.OutputParameter().n_slices;
    base5.Backward(base5.OutputParameter(), subError, base5.Delta());
    bias5.Backward(bias5.OutputParameter(), base5.Delta(), bias5.Delta());
    conv5.Backward(conv5.OutputParameter(), bias5.Delta(), conv5.Delta());
    baseProj5.Backward(baseProj5.OutputParameter(), conv5.Delta(), baseProj5.Delta());
    biasProj5.Backward(biasProj5.OutputParameter(), baseProj5.Delta(), biasProj5.Delta());
    proj5.Backward(proj5.OutputParameter(), biasProj5.Delta(), proj5.Delta());

    subError = error.slices(slice_idx, slice_idx + basePool.OutputParameter().n_slices - 1);
    slice_idx += basePool.OutputParameter().n_slices;
    basePool.Backward(basePool.OutputParameter(), subError, basePool.Delta());
    biasPool.Backward(biasPool.OutputParameter(), basePool.Delta(), biasPool.Delta());
    convPool.Backward(convPool.OutputParameter(), biasPool.Delta(), convPool.Delta());
    pool3.Backward(pool3.OutputParameter(), convPool.Delta(), pool3.Delta());
  }

  template<typename eT>
  void Gradient(const arma::Cube<eT>&, arma::Cube<eT>& delta, arma::Cube<eT>&)
  {
    size_t slice_idx = 0;
    arma::cube deltaNext = delta.slices(slice_idx, slice_idx + bias1.OutputParameter().n_slices - 1);
    conv1.Gradient(conv1.InputParameter(), bias1.Delta(), conv1.Gradient());
    bias1.Gradient(bias1.InputParameter(), deltaNext, bias1.Gradient());
    slice_idx += bias1.OutputParameter().n_slices;
       
    deltaNext = delta.slices(slice_idx, slice_idx + bias3.OutputParameter().n_slices - 1);
    proj3.Gradient(proj3.InputParameter(), biasProj3.Delta(), proj3.Gradient());
    biasProj3.Gradient(biasProj3.InputParameter(), conv3.Delta(), biasProj3.Gradient());
    conv3.Gradient(conv3.InputParameter(), bias3.Delta(), conv3.Gradient());
    bias3.Gradient(bias3.InputParameter(), deltaNext, bias3.Gradient());
    slice_idx += bias3.OutputParameter().n_slices;
   
    deltaNext = delta.slices(slice_idx, slice_idx + bias5.OutputParameter().n_slices - 1);     
    proj5.Gradient(proj5.InputParameter(), biasProj5.Delta(), proj5.Gradient());
    biasProj5.Gradient(biasProj5.InputParameter(), conv5.Delta(), biasProj5.Gradient());
    conv5.Gradient(conv5.InputParameter(), bias5.Delta(), conv5.Gradient());

    bias5.Gradient(bias5.InputParameter(), deltaNext, bias5.Gradient());
    slice_idx += bias5.OutputParameter().n_slices;
    

    deltaNext = delta.slices(slice_idx, slice_idx + biasPool.OutputParameter().n_slices - 1);
 
    convPool.Gradient(convPool.InputParameter(), biasPool.Delta(), convPool.Gradient());
    biasPool.Gradient(biasPool.InputParameter(), deltaNext, biasPool.Gradient());
    slice_idx += biasPool.OutputParameter().n_slices;
    
  }

  //! Get the weights.
  OutputDataType const& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;


}; // class InceptionLayer

} // namespace ann
} // namspace mlpack

#endif
