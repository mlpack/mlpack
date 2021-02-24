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
namespace ann { /** Artificial Neural Network. */

template<typename InputType, typename OutputType, int module>
Inception3<InputType, OutputType, module> :: Inception3(
    const size_t inSize,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t outA,
    const size_t outBOne,
    const size_t outBTwo,
    const size_t outCOne,
    const size_t outCTwo,
    const size_t outCThree,
    const size_t outCFour,
    const size_t outDOne,
    const size_t outDTwo,
    const size_t outDThree,
    const size_t outDFour,
    const size_t outDfive,
    const size_t outEOne,
    const size_t outETwo,
    const size_t outEThree,
    const size_t outFOne,
    const size_t outFTwo)
{
  //! Build the Inception3A module 
  if(module == 1)
  {
    model = new Concat(true);

    //! Build Network A
    Sequential networkA;
    networkA.Add<Convolution>(inSize, outA, 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkA.Add<BatchNorm>();
    networkA.Add<ReLULayer>();

    model->Add(networkA);

    //! Build Network B
    Sequential networkB;

    networkB.Add<Convolution>(inSize, outBOne, 1, 1, 1, 1, 0, 0, 
      inputWidth, inputHeight);
    networkB.Add<BatchNorm>();
    networkB.Add<ReLULayer>();

    networkB.Add<Convolution>(outBOne, outBTwo, 3, 3, 1, 1, 0, 0,
       inputWidth, inputHeight);
    networkB.Add<BatchNorm>();
    networkB.Add<ReLULayer>();

    model->Add(networkB);

    //! Build Network C
    Sequential networkC;

    networkC.Add<Convolution>(inSize, outCOne, 3, 3, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC.Add<BatchNorm>();
    networkC.Add<ReLULayer>();

    networkC.Add<Convolution>(outCOne, outCTwo, 3, 3, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkC.Add<BatchNorm>();
    networkC.Add<ReLULayer>();

    model->Add(networkC);

    //! Build Network D
    Sequential networkD;

    networkD.Add<MaxPool>(3, 3, 1, 1);
    
    networkD.Add<Convolution>(inSize, outDOne, 1, 1, 1, 1, 0, 0,
        inputWidth, inputHeight);
    networkD.Add<BatchNorm>();
    networkD.Add<ReLULayer>();

    model->Add(networkD);
  }
  //! Build Inception Block B module
  else if(module == 2)
  {
    //! TODO
  }
  //! Build Inception Block C module
  else if(module == 3)
  {
    //! TODO
  }
  //! Build Reduction Block A module
  else if(module == 4)
  {
    //! TODO
  }
  //! Build Reduction Block B module
  else if(module ==5)
  {
    //! TODO
  }
}

template<typename InputType, typename OutputType, int module>
Inception3<InputType, OutputType, module>::~Inception3()
{
  delete model;
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Forward(const InputType& input,
                                                    OutputType& output)
{
  model->Forward(std::move(input), std::move(output));
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Backward(const InputType& input,
                                                          const OutputType& gy,
                                                          OutputType &g)
{
  model->Backward(std::move(input), std::move(gy), std::move(g));
}

template<typename InputType, typename OutputType, int module>
void Inception3<InputType, OutputType, module>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  model->Gradient(std::move(input), std::move(error), std::move(gradient));
}

template<typename InputType, typename OutputType, int module>
template<typename Archive>
void Inception3<InputType, OutputType, module>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  // Don't know how this works..yet
}

} // namespace ann
} // namespace mlpack

#endif
}
}
