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

template<typename InputType, typename OutputType, int module>
Inception3<InputType, OutputType, module> :: Inception3(
    const size_t inSize,
    const size_t inputWidth,
    const size_t inputHeight,
    const arma::vec outA,
    const arma::vec outB,
    const arma::vec outC,
    const arma::vec outD)
{
  model = new Concat(true);

  //! Build the Inception3A module 
  if(module == 1)
  {
    //! Build Network A
    Sequential* networkA;
    networkA = new Sequential();

    networkA->Add<Convolution>(inSize, outA[0], 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkA->Add<BatchNorm>(outA[0]);
    networkA->Add<ReLULayer>();

    model->Add(networkA);

    //! Build Network B
    Sequential* networkB;
    networkB = new Sequential();

    networkB->Add<Convolution>(inSize, outB[0], 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[0]);
    networkB->Add<ReLULayer>();

    networkB->Add<Convolution>(outB[0], outB[1], 5, 5, 1, 1, 2, 2,
       inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[1]);
    networkB->Add<ReLULayer>();

    model->Add(networkB);

    //! Build Network C
    Sequential* networkC;
    networkC = new Sequential();

    networkC->Add<Convolution>(inSize, outC[0], 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[0]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[0], outC[1], 3, 3, 1, 1, 1, 1,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[1]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[1], outC[2], 3, 3, 1, 1, 1, 1,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[2]);
    networkC->Add<ReLULayer>();

    model->Add(networkC);

    //! Build Network D
    Sequential* networkD;
    networkD = new Sequential();

    networkD->Add<AdaptiveMeanPooling>(inputWidth, inputHeight);
    
    networkD->Add<Convolution>(inSize, outD[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[0]);
    networkD->Add<ReLULayer>();

    model->Add(networkD);
  }

  //! Build Inception B module
  else if(module == 2)
  {
    //! Build Network A
    Sequential* networkA;
    networkA = new Sequential();

    networkA->Add<MaxPooling>(3, 3, 2, 2);

    model->Add(networkA);

    //! Build Network B
    Sequential* networkB;
    networkB = new Sequential();

    networkB->Add<Convolution>(inSize, outB[0], 3, 3, 2, 2, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[0]);
    networkB->Add<ReLULayer>();

    model->Add(networkB);

    //! Build Network C
    Sequential* networkC;
    networkC = new Sequential();

    networkC->Add<Convolution>(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[0]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[0], outC[1], 3, 3, 1, 1, 1, 1,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[1]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[1], outC[2], 3, 3, 2, 2, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[2]);
    networkC->Add<ReLULayer>();

    model->Add(networkC);
  }

  //! Build Inception Block C module
  else if(module == 3)
  {
    //! Build Network A
    Sequential* networkA;
    networkA = new Sequential();

    networkA->Add<Convolution>(inSize, outA[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkA->Add<BatchNorm>(outA[0]);
    networkA->Add<ReLULayer>();

    model->Add(networkA);

    //! Build Network B
   Sequential* networkB;
    networkB = new Sequential();

    networkB->Add<AdaptiveMeanPooling>(inputWidth, inputHeight);

    networkB->Add<Convolution>(inSize, outB[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[0]);
    networkB->Add<ReLULayer>();

    model->Add(networkB);

    //! Build Network C
    Sequential* networkC;
    networkC = new Sequential();

    networkC->Add<Convolution>(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[0]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[0], outC[1], 1, 7, 1, 1, 0, 3,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[1]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[1], outC[2], 7, 1, 1, 1, 3, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[2]);
    networkC->Add<ReLULayer>();

    model->Add(networkC);

    //! Build Network D
    Sequential* networkD;
    networkD = new Sequential();

    networkD->Add<Convolution>(inSize, outD[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[0]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[0], outD[1], 7, 1, 1, 1, 3, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[1]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[1], outD[2], 1, 7, 1, 1, 0, 3,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[2]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[2], outD[3], 7, 1, 1, 1, 3, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[3]);
    networkD->Add<ReLULayer>();

    networkD->Add<Convolution>(outD[3], outD[4], 1, 7, 1, 1, 0, 3,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[4]);
    networkD->Add<ReLULayer>();

    model->Add(networkD);

  }
  //! Build Inception D module
  else if(module == 4)
  {
    //! Build Network A
    Sequential* networkA;
    networkA = new Sequential();

    networkA->Add<MaxPooling>(3, 3, 2, 2);

    model->Add(networkA);

    //! Build Network B
    Sequential* networkB;
    networkB = new Sequential();

    networkB->Add<Convolution>(inSize, outB[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkB->Add<BatchNorm>(outB[0]);
    networkB->Add<ReLULayer>();

    networkB->Add<Convolution>(outB[0], outB[1], 3, 3, 2, 2, 0, 0,
        inputWidth, inputHeight);                                    
    networkB->Add<BatchNorm>(outB[1]);
    networkB->Add<ReLULayer>();                                

    model->Add(networkB);

    //! Build Network C
    Sequential* networkC;
    networkC = new Sequential();

    networkC->Add<Convolution>(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[0]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[0], outC[1], 1, 7, 1, 1, 0, 3,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[1]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[1], outC[2], 7, 1, 1, 1, 3, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[2]);
    networkC->Add<ReLULayer>();

    networkC->Add<Convolution>(outC[2], outC[3], 3, 3, 2, 2, 0, 0,
        inputWidth, inputHeight);
    networkC->Add<BatchNorm>(outC[3]);
    networkC->Add<ReLULayer>();

    model->Add(networkC);

  }
  //! Build Inception E module
  else if(module == 5)
  {
    //! Build Network A
    Sequential* networkA; 
    networkA = new Sequential();

    networkA->Add<Convolution>(inSize, outA[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkA->Add<BatchNorm>(outA[0]);
    networkA->Add<ReLULayer>();

    model->Add(networkA);

    //! Build Network B
    Concat* networkB;
    networkB = new Concat(true);

    Sequential* branch1x1B;
    branch1x1B = new Sequential();

    branch1x1B->Add<Convolution>(inSize, outB[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    branch1x1B->Add<BatchNorm>(outB[0]);
    branch1x1B->Add<ReLULayer>();

    Sequential* branch1x3B;
    branch1x3B = new Sequential();

    branch1x3B->Add(branch1x1B);

    branch1x3B->Add<Convolution>(outB[0], outB[1], 1, 3, 1, 1, 0, 1,
        inputWidth, inputHeight);
    branch1x3B->Add<BatchNorm>(outB[1]);
    branch1x3B->Add<ReLULayer>();

    Sequential* branch3x1B;
    branch3x1B = new Sequential();

    branch3x1B->Add(branch1x1B);

    branch3x1B->Add<Convolution>(outB[0], outB[2], 3, 1, 1, 1, 1, 0,
        inputWidth, inputHeight);
    branch3x1B->Add<BatchNorm>(outB[2]);
    branch3x1B->Add<ReLULayer>();

    networkB->Add(branch1x3B);
    networkB->Add(branch3x1B);

    model->Add(networkB);

    //! Build Network C
    Concat* networkC;
    networkC = new Concat(true);

    Sequential* branch1x1C;
    branch1x1C = new Sequential();

    branch1x1C->Add<Convolution>(inSize, outC[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    branch1x1C->Add<BatchNorm>(outC[0]);
    branch1x1C->Add<ReLULayer>();

    Sequential* branch3x3C;
    branch3x3C = new Sequential();

    branch3x3C->Add<Convolution>(outC[0], outC[1], 3, 3, 1, 1, 1, 1,
        inputWidth, inputHeight);
    branch3x3C->Add<BatchNorm>(outC[1]);
    branch3x3C->Add<ReLULayer>();

    Sequential* branch3x1C;
    branch3x1C = new Sequential();
    
    branch3x1C->Add(branch1x1C);
    branch3x1C->Add(branch3x3C);

    branch3x1C->Add<Convolution>(outC[1], outC[2], 3, 1, 1, 1, 1, 0,
        inputWidth, inputHeight);
    branch3x1C->Add<BatchNorm>(outC[2]);
    branch3x1C->Add<ReLULayer>();

    Sequential* branch1x3C;
    branch1x3C = new Sequential();

    branch1x3C->Add(branch1x1C);
    branch1x3C->Add(branch3x3C);

   branch1x3C->Add<Convolution>(outC[1], outC[3], 1, 3, 1, 1, 0, 1,
        inputWidth, inputHeight);
    branch1x3C->Add<BatchNorm>(outC[3]);
    branch1x3C->Add<ReLULayer>();

    networkC->Add(branch3x1C);
    networkC->Add(branch1x3C);

    model->Add(networkC);

    //! Build Network D
    Sequential* networkD;
    networkD = new Sequential();

    networkD->Add<AdaptiveMeanPooling>(inputWidth, inputHeight);

    networkD->Add<Convolution>(inSize, outD[0], 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD->Add<BatchNorm>(outD[0]);
    networkD->Add<ReLULayer>();

    model->Add(networkD);
  }
}

template<typename InputType, typename OutputType, int module>
Inception3<InputType, OutputType, module>::~Inception3()
{
  delete model;
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Reset()
{
  model->Reset();
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Forward(const InputType& input,
                                                    OutputType& output)
{
  model->Forward(std::move(input), output);
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Backward(const InputType& input,
                                                          const OutputType& gy,
                                                          OutputType &g)
{
  model->Backward(std::move(input), std::move(gy), g);
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  model->Gradient(std::move(input), std::move(error), gradient);
}

template<typename InputType, typename OutputType, int module>
template<typename Archive>
void Inception3<InputType, OutputType, module>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  //Not sure how to do this..
}

} // namespace ann
} // namespace mlpack

#endif
