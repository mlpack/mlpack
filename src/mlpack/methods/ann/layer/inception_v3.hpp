/**
 * @file inception_v3.hpp
 * @author Shah Anwaar Khalid
 *
 * Definition of Inception V3 layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_INCEPTION_V3_HPP
#define MLPACK_METHODS_ANN_LAYER_INCEPTION_V3_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {
namespace ann /**Artificial Neural Network. */

/**
 * Declaration of the Inception V3 layer class. This class will be a
 * wrapper around concat layer object. It also uses sequential layers
 * for it's implementation
 *
 * The Inception V3 model consists of the following 5 modules:
 *
 * 1. Inception Block A:
 * Concat (
 * Network A : 1 x 1 convolution
 * Network B : 1 x 1 followed by 3 x 3 convolutions
 * Network C : 3 x 3 convolution followed by another 3 x 3 convolution
 * Network D : 3 x 3 max pool followed by 1 x 1 convolutions
 * )
 *
 * 2. Reduction Block A:
 * Concat (
 * Network A : 3 x 3 Max Pool
 * Network B : 3 x 3 convolution
 * Network C : 1 x 1 convolution followed by two 3 x 3 convolutions
 * )
 *
 * 3. Inception Block B:
 * Concat (
 * Network A : 1 x 1 convolution
 * Network B : 3 x 3 Avg Pool followed by 1 x 1 convolution
 * Network C : 1 x 1 convolution followed by 1 x 7 & 7 x 1 convolution
 * Network D : 1 x 1 convolution followed by 2 blocks of 7 X 1 & 1 x 7 convolutions
 * )
 *
 * 4. Reduction Block B:
 * Concat (
 * Network A : 3 x 3 Max pool
 * Network B : 1 x 1 convolution followed by 3 x 3 convolution
 * Network C : 1 x 1 convolution followed by 1 x 7 and
 *             7 x 1 convolution followed by another 3x3 convolution
 * )
 *
 * 5. Inception Block C:
 * Concat (
 * Network A : 1 x 1 convolution
 * Network B : 1 x 1 convolution followed by 1 x 3 convolution
 * Network C : 1 x 1 convolution followed by 3 x 1 convolution
 * Network D : 1 x 1 convolution followed by 3 x 3 & 3 x 1 convolution
 * Network E : 1 x 1 convolution followed by 3 x 3  & 1 x 3 convoltion
 * Network F : 3 x 3 max pooling followed by 1 x 1 convolution
 * )
 *
 * For more information , see the following paper:
 * @code
 * @article {
 * author = { Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe,
 *            Jonathon Shlens},
 * title  = { Rethinking the Inception Architecture for Computer Vision },
 * year   = { 2015 },
 * url    = { https://arxiv.org/pdf/1512.00567v3.pdf }
 * }
 * @endcode
 *
 * @tparam InputType The type of layer's inputs. The layer automatically casts
 *  inputs to this type ( Default: arma::mat}
 * @tparam OutputType The type of the computation which also causes the output
 *  to also be in this type. The type also allows the computation and weight
 *  type to differ from input type (Default: arma::mat)
 * @tparam module The type of Inception Block
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat,
    int module = 1
>
class Inception3 : public Layer <InputType, OutputType>
{
  public:

  /**
   * Create the Inception3 Object.
   *
   * @param inSize The number of input maps.
   * @param inputWidth The width of input data.
   * @param inputHeight The height of input data.
   * @param outA The number of output maps of network A.
   * @param outBOne The number of output maps of the first layer of network B.
   * @param outBTwo The number of output maps of the second layer of network B.
   * @param outBTwo The number of output maps of the second layer of network B.
   * @param outCOne The number of output maps of the first layer of network C.
   * @param outCTwo The number of output maps of the second layer of network C.
   * @param outCThree The number of output maps of the third layer of network C.
   * @param outCFour The number of output maps of the fourth layer of network C.
   * @param outDOne The number of output maps of the first layer of network D.
   * @param outDTwo The number of output maps of the second layer of network D.
   * @param outDThree The number of output maps of the third layer of network D.
   * @param outDFour The number of output maps of the fourth layer of network D.
   * @param outDFive The number of output maps of the fifth layer of network D.
   * @param outEOne The number of output maps of the first layer of network E.
   * @param outETwo The number of output maps of the second layer of network E.
   * @param outEThree The number of output maps of the third layer of network E.
   * @param outFOne The number of output maps of the first layer of network F.
   * @param outFTwo The number of output maps of the second layer of network F.
   */

  Inception3(const size_t inSize,
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
             const size_t outDFive,
             const size_t outEOne,
             const size_t outETwo,
             const size_t outEThree,
             const size_t outFOne,
             const size_t outFTwo);

  //! Destructor to release allocated memory.
  ~Inception3();

  /**
   * Forward pass of the Inception3 Layer.
   * It calls the Forward() function of the concat layer object
   *
   * @param input Input Data for the layer.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Backward pass through the layer. This function calls the Backwar()
   * function of the concat layer object.
   *
   * @param input The input activations.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input,
                const OutputType& gy,
                OutputType &g);

  /**
   * Caculate the gradients. This will also call the gradient function
   * of the concat layer object
   *
   * @param input The input activations.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */

  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get delta
  OutputType const& Delta() const { return model->Delta();}
  //! Modify delta
  OutputType& Delta() { return model->Delta(); }

  //! Get gradients
  OutputType const& Gradient() const { return model->Gradient(); }
  //! Modify gradients
  OutputType& Gradient() { return model->Gradient(); }

  //! Input Parameter
  InputType const& InputParameter() const { return model->InputParameter(); }
  //! Modify the input parameter
  InputType& InputParameter() { return model->InputParameter(); }

  //! Return the model modules
  std::vector<Layer<InputType, OutputType>*>& Model(){ return model->Model(); }

  //! Get the output parameter
  OutputType const& OutputParameter() const
  {
    return model->OutputParameter();
  }

  //! Modify the output parameter
  OutputType& OutputParameter()
  {
    return model->OutputParameter();
  }
  //! Get the parameters.
  OutputDataType const& Parameters() const { return model->Parameters(); }
  //! Modify the parameters.
  OutputDataType& Parameters() { return model->Parameters(); }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored concat layer object.
  Concat<InputType, OutputType>* model;
};
  //! Typedef for various blocks of the model
  using Inception3A = Inception3< InputType , OutputType, 1>;
  using Inception3B = Inception3< InputType , OutputType, 2>;
  using Inception3C = Inception3< InputType , OutputType, 3>;
  using Reduction3A = Inception3< InputType , OutputType, 4>;
  using Reduction3B = Inception3< InputType , OutputType, 5>;


} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "inception_v3_impl.hpp"

#endif
}
}
