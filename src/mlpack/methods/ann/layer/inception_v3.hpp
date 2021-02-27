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
namespace ann { /**Artificial Neural Network. */

/**
 * Declaration of the Inception V3 layer class. This class will be a
 * wrapper around concat layer object. It also uses sequential layers
 * for it's implementation
 *
 * The Inception V3 model consists of the following 5 modules:
 *
 * 1. Inception Module A:
 * Concat (
 * Network A : 1 x 1 convolution
 * Network B : 1 x 1 followed by 3 x 3 convolution
 * Network C : 3 x 3 convolution followed by another 3 x 3 convolution
 * Network D : 3 x 3 max pool followed by 1 x 1 convolution
 * )
 *
 * 2. Reduction Module A:
 * Concat (
 * Network A : 3 x 3 max pool
 * Network B : 3 x 3 convolution
 * Network C : 1 x 1 convolution followed by two 3 x 3 convolution
 * )
 *
 * 3. Inception Module B:
 * Concat (
 * Network A : 1 x 1 convolution
 * Network B : 3 x 3 avg pool followed by 1 x 1 convolution
 * Network C : 1 x 1 convolution followed by 1 x 7 & 7 x 1 convolution
 * Network D : 1 x 1 convolution followed by 2 blocks of 7 X 1 & 1 x 7 convolution
 * )
 *
 * 4. Reduction Module B:
 * Concat (
 * Network A : 3 x 3 max pool
 * Network B : 1 x 1 convolution followed by 3 x 3 convolution
 * Network C : 1 x 1 convolution followed by 1 x 7 and
 *             7 x 1 convolution followed by 3x3 convolution
 * )
 *
 * 5. Inception Module C:
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
class Inception3 : public Layer<InputType, OutputType>
{
  public:

  /**
   * Create the Inception3 Object.
   *
   * @param inSize The number of input maps.
   * @param inputWidth The width of input data.
   * @param inputHeight The height of input data.
   * @param outA The number of output maps of all layers of network A.
   * @param outB The number of output maps of all layers of network B.
   * @param outC The number of output maps of all layers of network C.
   * @param outD The number of output maps of all layers of network D.
   * @param outE The number of output maps of all layers of network E.
   * @param outF The number of output maps of all layers of network F.
   */

  Inception3(const size_t inSize,
             const size_t inputWidth,
             const size_t inputHeight,
             const arma::vec outA,
             const arma::vec outB,
             const arma::vec outC,
             const arma::vec outD,
             const arma::vec outE,
             const arma::vec outF);

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
  using Inception3A = Inception3< arma::mat, arma::mat, 1>;
  using Inception3B = Inception3< arma::mat, arma::mat, 2>;
  using Inception3C = Inception3< arma::mat, arma::mat, 3>;
  using Reduction3A = Inception3< arma::mat, arma::mat, 4>;
  using Reduction3B = Inception3< arma::mat, arma::mat, 5>;


} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "inception_v3_impl.hpp"

#endif
