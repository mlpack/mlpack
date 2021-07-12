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
#include "layer_types.hpp"

namespace mlpack {
namespace ann /**Artificial Neural Network. */{

/**
 * Declaration of the Inception V3 layer class. This class will be a
 * wrapper around concat layer object. It also uses sequential layers
 * for it's implementation
 *
 * The Inception V3 layer consists of the following 5 modules:
 *
 * NOTE: Each convolution block consists of a convolutional layer
 * followed by BatchNorm and ReLU layers.
 *
 * 1. Inception Module A:
 * Concat (
 * Network A : 1 x 1 convolution block.
 * Network B : 1 x 1 followed by 5 x 5 convolution block.
 * Network C : 1 x 1 convolution block followed by two 3 x 3 convolution blocks.
 * Network D : 3 x 3 avg pool followed by 1 x 1 convolution block.
 * )
 *
 * 2. Inception Module B:
 * Concat (
 * Network A : 3 x 3 max pool.
 * Network B : 3 x 3 convolution block.
 * Network C : 1 x 1 convolution block followed by two 3 x 3 convolution blocks.
 * )
 *
 * 3. Inception Module C:
 * Concat (
 * Network A : 1 x 1 convolution block.
 * Network B : 3 x 3 avg pool followed by 1 x 1 convolution block.
 * Network C : 1 x 1 convolution block followed by 1 x 7 & 7 x 1 convolution blocks.
 * Network D : 1 x 1 convolution block followed by
 *             2 blocks of 7 X 1 & 1 x 7 convolution blocks.
 * )
 *
 * 4. Inception Module D:
 * Concat (
 * Network A : 3 x 3 max pool
 * Network B : 1 x 1 convolution block followed by 3 x 3 convolution block.
 * Network C : 1 x 1 convolution block followed by 1 x 7 and
 *             7 x 1 convolution blocks followed by 3x3 convolution block.
 * )
 *
 * 5. Inception Module E:
 * Concat (
 * Network A : 1 x 1 convolution block.
 * Network B : Concat(
 *                                              |-- 1 x 3 convolution block 
 *                    1 x 1 convolution block --|
 *                                              |-- 3 x 1 convolution block  
 *                   )
 * Network C :Concat(
 *                                                      |-- 1 x 3 convolution block      
 *                   1 x 1 -- 3 x 3 convolution block --|
 *                                                      |-- 3 x 1 convolution block
 *                  )
 * Network D : 3 x 3 avg pool followed by 1 x 1 convolution block.
 * )
 *
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
 * @tparam InputDataType The type of layer's inputs. The layer automatically casts
 *  inputs to this type ( Default: arma::mat}
 * @tparam OutputDataType The type of the computation which also causes the output
 *  to also be in this type. The type also allows the computation and weight
 *  type to differ from input type (Default: arma::mat)
 * @tparam module The type of Inception Block
 */
template<
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    int module = 1,
    typename... CustomLayers
>
class Inception3
{
  public:
   //! Create the Inception3 object
   Inception3();

   /**
    * Create the Inception3 Object.
    *
    * @param inSize The number of input maps.
    * @param inputWidth The width of input data.
    * @param inputHeight The height of input data.
    * @param outA Vector of output maps of all layers of network A.
    * @param outB Vector of output maps of all layers of network B.
    * @param outC Vector of output maps of all layers of network C.
    * @param outD Vector of output maps of all layers of network D.
    */

   Inception3(const size_t inSize,
              const size_t inputWidth,
              const size_t inputHeight,
              const arma::vec outA,
              const arma::vec outB,
              const arma::vec outC,
              const arma::vec outD);

   //! Destructor to release allocated memory.
   ~Inception3();

   /**
    * Reset the layer parameters. This method is called to assign
    * the allocated memory to the internal learnable parameters.
    */
   void Reset();

   /**
    * Forward pass of the Inception3 Layer.
    * It calls the Forward() function of the concat layer object
    *
    * @param input Input Data for the layer.
    * @param output Resulting output activation.
    */
   template<typename eT>
   void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

   /**
    * Backward pass through the layer. This function calls the Backwar()
    * function of the concat layer object.
    *
    * @param input The input activations.
    * @param gy The backpropagated error.
    * @param g The calculated gradient.
    */
   template<typename eT>
   void Backward(const arma::Mat<eT>& input,
                 const arma::Mat<eT>& gy,
                 arma::Mat<eT>& g);

   /**
    * Caculate the gradients. This will also call the gradient function
    * of the concat layer object
    *
    * @param input The input activations.
    * @param error The calculated error.
    * @param gradient The calculated gradient.
    */

   template<typename eT>
   void Gradient(const arma::Mat<eT>& input,
                 const arma::Mat<eT>& error,
                 arma::Mat<eT>& gradient);

   //! Get delta
   OutputDataType const& Delta() const { return layer->Delta();}
   //! Modify delta
   OutputDataType& Delta() { return layer->Delta(); }

   //! Get gradients
   OutputDataType const& Gradient() const { return layer->Gradient(); }
   //! Modify gradients
   OutputDataType& Gradient() { return layer->Gradient(); }

   //! Input Parameter
   InputDataType const& InputParameter() const 
   { 
     return layer->InputParameter();
   }
   //! Modify the input parameter
   InputDataType& InputParameter() { return layer->InputParameter(); }

   //! Return the layer modules
   std::vector<LayerTypes<CustomLayers...> >& Model(){ return layer->Model(); }

   //! Get the output parameter
   OutputDataType const& OutputParameter() const
   {
     return layer->OutputParameter();
   }

   //! Modify the output parameter
   OutputDataType& OutputParameter()
   {
     return layer->OutputParameter();
   }
   //! Get the parameters.
   OutputDataType const& Parameters() const { return layer->Parameters(); }
   //! Modify the parameters.
   OutputDataType& Parameters() { return layer->Parameters(); }

   /**
    * Serialize the layer.
    */
   template<typename Archive>
   void serialize(Archive& ar, const unsigned int /* version */);

  private:
   //! Locally-stored concat layer object.
   Concat<InputDataType, OutputDataType>* layer;
};
  //! Typedef for various blocks of the layer
  using Inception3A = Inception3< arma::mat, arma::mat, 1>;
  using Inception3B = Inception3< arma::mat, arma::mat, 2>;
  using Inception3C = Inception3< arma::mat, arma::mat, 3>;
  using Inception3D = Inception3< arma::mat, arma::mat, 4>;
  using Inception3E = Inception3< arma::mat, arma::mat, 5>;


} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "inception_v3_impl.hpp"

#endif
