/**
 * @file inception_v3_impl.hpp
 * @author Shah Anwaar Khalid
 *
 * Implementation of the InceptionV3 Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it uder the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_INCEPTION_V3_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_INCEPTION_V3_IMPL_HPP

// In case it is not included.
#include "inception_v3.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType, int module,
      typename... CustomLayers>
Inception3<InputDataType, OutputDataType, module, CustomLayers...>::Inception3()
{
  // Nothing to do here
}

template<typename InputDataType, typename OutputDataType, int module,
      typename... CustomLayers>
Inception3<InputDataType, OutputDataType, module, CustomLayers...>::Inception3(
    const size_t inSize,
    const size_t inputWidth,
    const size_t inputHeight,
    const arma::vec outA,
    const arma::vec outB,
    const arma::vec outC,
    const arma::vec outD)
{
  layer= new Concat<>(true);

  //! Build the Inception3A module 
  if(module == 1)
  {
    //! Build Network A
    Sequential<> *networkA;
    networkA = new Sequential<>(true);

    networkA->Add<Convolution<> >(inSize, outA[0], 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkA->Add<BatchNorm<> >(outA[0]);
    networkA->Add<ReLULayer<> >();

    layer->Add(networkA);

    //! Build Network B
    Sequential<> *networkB;
    networkB = new Sequential<>(true);

    networkB->Add<Convolution<> >(inSize, outB[0], 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkB->Add<BatchNorm<> >(outB[0]);
    networkB->Add<ReLULayer<> >();

    networkB->Add<Convolution<> >(outB[0], outB[1], 5, 5, 1, 1, 2, 2,
       inputWidth, inputHeight);
    networkB->Add<BatchNorm<> >(outB[1]);
    networkB->Add<ReLULayer<> >();

    layer->Add(networkB);

    //! Build Network C
    Sequential<> *networkC;
    networkC = new Sequential<>(true);

    networkC->Add<Convolution<> >(inSize, outC[0], 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[0]);
    networkC->Add<ReLULayer<> >();

    networkC->Add<Convolution<> >(outC[0], outC[1], 3, 3, 1, 1, 1, 1,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[1]);
    networkC->Add<ReLULayer<> >();

    networkC->Add<Convolution<> >(outC[1], outC[2], 3, 3, 1, 1, 1, 1,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[2]);
    networkC->Add<ReLULayer<> >();

    layer->Add(networkC);

    //! Build Network D
    Sequential<> *networkD;
    networkD = new Sequential<>(true);

    networkD->Add<AdaptiveMeanPooling<> >(inputWidth, inputHeight);
    
    networkD->Add<Convolution<> >(inSize, outD[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm<> >(outD[0]);
    networkD->Add<ReLULayer<> >();

    layer->Add(networkD);
  }

  //! Build Inception B module
  else if(module == 2)
  {
    //! Build Network A
    Sequential<>* networkA;
    networkA = new Sequential<>(true);

    networkA->Add<MaxPooling<> >(3, 3, 2, 2);

    layer->Add(networkA);

    //! Build Network B
    Sequential<>* networkB;
    networkB = new Sequential<>(true);

    networkB->Add<Convolution<> >(inSize, outB[0], 3, 3, 2, 2, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm<> >(outB[0]);
    networkB->Add<ReLULayer<> >();

    layer->Add(networkB);

    //! Build Network C
    Sequential<> *networkC;
    networkC = new Sequential<>(true);

    networkC->Add<Convolution<> >(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[0]);
    networkC->Add<ReLULayer<> >();

    networkC->Add<Convolution<> >(outC[0], outC[1], 3, 3, 1, 1, 1, 1,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[1]);
    networkC->Add<ReLULayer<> >();

    networkC->Add<Convolution<> >(outC[1], outC[2], 3, 3, 2, 2, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[2]);
    networkC->Add<ReLULayer<> >();

    layer->Add(networkC);
  }

  //! Build Inception Block C module
  else if(module == 3)
  {
    //! Build Network A
    Sequential<> *networkA;
    networkA = new Sequential<>(true);

    networkA->Add<Convolution<> >(inSize, outA[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkA->Add<BatchNorm<> >(outA[0]);
    networkA->Add<ReLULayer<> >();

    layer->Add(networkA);

    //! Build Network B
   Sequential<> *networkB;
   networkB = new Sequential<>(true);

    networkB->Add<AdaptiveMeanPooling<> >(inputWidth, inputHeight);

    networkB->Add<Convolution<> >(inSize, outB[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm<> >(outB[0]);
    networkB->Add<ReLULayer<> >();

    layer->Add(networkB);

    //! Build Network C
    Sequential<> *networkC;
    networkC = new Sequential<>(true);

    networkC->Add<Convolution<> >(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[0]);
    networkC->Add<ReLULayer<> >();

    networkC->Add<Convolution<> >(outC[0], outC[1], 1, 7, 1, 1, 0, 3,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[1]);
    networkC->Add<ReLULayer<> >();

    networkC->Add<Convolution<> >(outC[1], outC[2], 7, 1, 1, 1, 3, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[2]);
    networkC->Add<ReLULayer<> >();

    layer->Add(networkC);

    //! Build Network D
    Sequential<> *networkD;
    networkD = new Sequential<>(true);

    networkD->Add<Convolution<> >(inSize, outD[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm<> >(outD[0]);
    networkD->Add<ReLULayer<> >();

    networkD->Add<Convolution<> >(outD[0], outD[1], 7, 1, 1, 1, 3, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm<> >(outD[1]);
    networkD->Add<ReLULayer<> >();

    networkD->Add<Convolution<> >(outD[1], outD[2], 1, 7, 1, 1, 0, 3,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm<> >(outD[2]);
    networkD->Add<ReLULayer<> >();

    networkD->Add<Convolution<> >(outD[2], outD[3], 7, 1, 1, 1, 3, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm<> >(outD[3]);
    networkD->Add<ReLULayer<> >();

    networkD->Add<Convolution<> >(outD[3], outD[4], 1, 7, 1, 1, 0, 3,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm<> >(outD[4]);
    networkD->Add<ReLULayer<> >();

    layer->Add(networkD);

  }
  //! Build Inception D module
  else if(module == 4)
  {
    //! Build Network A
    Sequential<> *networkA;
    networkA = new Sequential<>(true);

    networkA->Add<MaxPooling<> >(3, 3, 2, 2);

    layer->Add(networkA);

    //! Build Network B
    Sequential<> *networkB;
    networkB = new Sequential<>(true);

    networkB->Add<Convolution<> >(inSize, outB[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm<> >(outB[0]);
    networkB->Add<ReLULayer<> >();

    networkB->Add<Convolution<> >(outB[0], outB[1], 3, 3, 2, 2, 0, 0,
        inputWidth, inputHeight);                                    
    networkB->Add<BatchNorm<> >(outB[1]);
    networkB->Add<ReLULayer<> >();                                

    layer->Add(networkB);

    //! Build Network C
    Sequential<> *networkC;
    networkC = new Sequential<>(true);

    networkC->Add<Convolution<> >(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[0]);
    networkC->Add<ReLULayer<> >();

    networkC->Add<Convolution<> >(outC[0], outC[1], 1, 7, 1, 1, 0, 3,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[1]);
    networkC->Add<ReLULayer<> >();

    networkC->Add<Convolution<> >(outC[1], outC[2], 7, 1, 1, 1, 3, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[2]);
    networkC->Add<ReLULayer<> >();

    networkC->Add<Convolution<> >(outC[2], outC[3], 3, 3, 2, 2, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm<> >(outC[3]);
    networkC->Add<ReLULayer<> >();

    layer->Add(networkC);

  }
  //! Build Inception E module
  else if(module == 5)
  {
    //! Build Network A
    Sequential<> *networkA; 
    networkA = new Sequential<>(true);

    networkA->Add<Convolution<> >(inSize, outA[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkA->Add<BatchNorm<> >(outA[0]);
    networkA->Add<ReLULayer<> >();

    layer->Add(networkA);

    //! Build Network B
    Concat<> *networkB;
    networkB = new Concat<>(true);

    Sequential<> *branch1x1B;
    branch1x1B = new Sequential<>(true);

    branch1x1B->Add<Convolution<> >(inSize, outB[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    branch1x1B->Add<BatchNorm<> >(outB[0]);
    branch1x1B->Add<ReLULayer<> >();

    Sequential<> *branch1x3B;
    branch1x3B = new Sequential<>(true);

    branch1x3B->Add(branch1x1B);
    branch1x3B->Add<Convolution<> >(outB[0], outB[1], 1, 3, 1, 1, 0, 1,
        inputWidth, inputHeight);
    branch1x3B->Add<BatchNorm<> >(outB[1]);
    branch1x3B->Add<ReLULayer<> >();

    Sequential<> *branch3x1B;
    branch3x1B = new Sequential<>(true);

    branch3x1B->Add(branch1x1B);
    branch3x1B->Add<Convolution<> >(outB[0], outB[2], 3, 1, 1, 1, 1, 0,
        inputWidth, inputHeight);
    branch3x1B->Add<BatchNorm<> >(outB[2]);
    branch3x1B->Add<ReLULayer<> >();

    networkB->Add(branch1x3B);
    networkB->Add(branch3x1B);

    layer->Add(networkB);

    //! Build Network C
    Concat<> *networkC;
    networkC = new Concat<>(true);

    Sequential<> *branch1x1C;
    branch1x1C = new Sequential<>(true);

    branch1x1C->Add<Convolution<> >(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    branch1x1C->Add<BatchNorm<> >(outC[0]);
    branch1x1C->Add<ReLULayer<> >();

    Sequential<> *branch3x3C;
    branch3x3C = new Sequential<>(true);

    branch3x3C->Add<Convolution<> >(outC[0], outC[1], 3, 3, 1, 1, 1, 1,
        inputWidth, inputHeight);
    branch3x3C->Add<BatchNorm<> >(outC[1]);
    branch3x3C->Add<ReLULayer<> >();

    Sequential<> *branch3x1C;
    branch3x1C = new Sequential<>(true);

    branch3x1C->Add(branch1x1C);
    branch3x1C->Add(branch3x3C);
    branch3x1C->Add<Convolution<> >(outC[1], outC[2], 3, 1, 1, 1, 1, 0,
        inputWidth, inputHeight);
    branch3x1C->Add<BatchNorm<> >(outC[2]);
    branch3x1C->Add<ReLULayer<> >();

    Sequential<> *branch1x3C;
    branch1x3C = new Sequential<>(true);

    branch1x3C->Add(branch1x1C);
    branch1x3C->Add(branch3x3C);
    branch1x3C->Add<Convolution<> >(outC[1], outC[3], 1, 3, 1, 1, 0, 1,
        inputWidth, inputHeight);
    branch1x3C->Add<BatchNorm<> >(outC[3]);
    branch1x3C->Add<ReLULayer<> >();

    networkC->Add(branch3x1C);
    networkC->Add(branch1x3C);

    layer->Add(networkC);

  //! Build Network D
    Sequential<> *networkD;
    networkD = new Sequential<>(true);

    networkD->Add<AdaptiveMeanPooling<> >(inputWidth, inputHeight);

    networkD->Add<Convolution<> >(inSize, outD[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm<> >(outD[0]);
    networkD->Add<ReLULayer<> >();

    layer->Add(networkD);
  }
}

template<typename InputDataType, typename OutputDataType, int module,
      typename... CustomLayers>
Inception3<InputDataType, OutputDataType, module, CustomLayers...>::~Inception3()
{
  delete layer;
}

template<typename InputDataType, typename OutputDataType, int module,
  typename... CustomLayers>
void Inception3<InputDataType, OutputDataType, module, CustomLayers...>::Reset()
{
  layer->Reset();
}

template<typename InputDataType, typename OutputDataType, int module,
      typename... CustomLayers>
template<typename eT>
void Inception3<InputDataType, OutputDataType, module,
     CustomLayers...>::Forward(const arma::Mat<eT>& input,
                              arma::Mat<eT>& output)
{
  layer->Forward(std::move(input), output);
}

template<typename InputDataType, typename OutputDataType, int module,
    typename... CustomLayers>
template<typename eT>
void Inception3<InputDataType, OutputDataType, module,
      CustomLayers... >::Backward(const arma::Mat<eT>& input,
                                  const arma::Mat<eT>& gy,
                                  arma::Mat<eT>& g)
{
  layer->Backward(std::move(input), std::move(gy), g);
}

template<typename InputDataType, typename OutputDataType, int module,
  typename... CustomLayers>
template<typename eT>
void Inception3<InputDataType, OutputDataType, module,
      CustomLayers... >::Gradient(
      const arma::Mat<eT>& input,
      const arma::Mat<eT>& error,
      arma::Mat<eT>& gradient)
{
  layer->Gradient(std::move(input), std::move(error), gradient);
}

template<typename InputDataType, typename OutputDataType, int module,
  typename... CustomLayers>
template<typename Archive>
void Inception3<InputDataType, OutputDataType, module,
     CustomLayers...>::serialize(
    Archive& ar, const unsigned int /* version */)
{
}

} // namespace ann
} // namespace mlpack

#endif
