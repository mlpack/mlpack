/**
 * @file layer_names.hpp
 * @author Sreenik Seal
 *
 * Implementation of a class that converts a given ann layer to string format.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <boost/variant/static_visitor.hpp>
#include <iostream>
#include <string>

using namespace mlpack::ann;

class TestVisitor : public boost::static_visitor<std::string>
{
 public:
  TestVisitor() { /*Nothing to do here*/ }

  template<typename LayerType>
  std::string operator()(LayerType* layer) const
  {
    return LayerString(layer);
  }

 private:
  template<typename T>
  typename std::enable_if<
      std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "linear"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "convolution"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "relu"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "maxpooling"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "atrousconvolution"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "alphadropout"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "batchnorm"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "constant"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "dropconnect"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "dropout"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "layernorm"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "linearnobias"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "meanpooling"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "multiplyconstant"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "transposedconvolution"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "identity"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "tanh"; }
  
  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "elu"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "hardtanh"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "leakyrelu"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "prelu"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "sigmoid"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "logsoftmax"; }

  template<typename T>
  typename std::enable_if<
      !std::is_same<T, Linear<> >::value &&
      !std::is_same<T, Convolution<> >::value &&
      !std::is_same<T, ReLULayer<> >::value &&
      !std::is_same<T, MaxPooling<> >::value &&
      !std::is_same<T, AtrousConvolution<> >::value &&
      !std::is_same<T, AlphaDropout<> >::value &&
      !std::is_same<T, BatchNorm<> >::value &&
      !std::is_same<T, Constant<> >::value &&
      !std::is_same<T, DropConnect<> >::value &&
      !std::is_same<T, Dropout<> >::value &&
      !std::is_same<T, LayerNorm<> >::value &&
      !std::is_same<T, LinearNoBias<> >::value &&
      !std::is_same<T, MeanPooling<> >::value &&
      !std::is_same<T, MultiplyConstant<> >::value &&
      !std::is_same<T, TransposedConvolution<> >::value &&
      !std::is_same<T, IdentityLayer<> >::value &&
      !std::is_same<T, TanHLayer<> >::value &&
      !std::is_same<T, ELU<> >::value &&
      !std::is_same<T, HardTanH<> >::value &&
      !std::is_same<T, LeakyReLU<> >::value &&
      !std::is_same<T, PReLU<> >::value &&
      !std::is_same<T, SigmoidLayer<> >::value &&
      !std::is_same<T, LogSoftMax<> >::value,
      std::string>::type
  LayerString(T* layer) const { return "Unsupported"; }
};