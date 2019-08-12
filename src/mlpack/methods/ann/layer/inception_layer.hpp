/**
 * @file inception_layer.hpp
 * @author Marcus Edel
 * @author Toshal Agrawal
 *
 * Definition of the InceptionLayer layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_INCEPTION_LAYER_HPP

#include <mlpack/prereqs.hpp>
#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Declaration of the InceptionLayer layer class. The layer is the important
 * module of GoogLeNet.
 *
 * This class will be a wrapper around concat layer. It will create, modify and
 * delete contents of a concat layer object. It also uses sequential layers for
 * it's implementation.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @inproceedings{
 *   author  = {Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet,
 *             Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke,
 *             Andrew Rabinovich},
 *   title   = {Going deeper with convolutions},
 *   journal = {Proceedings of the IEEE Conference on Computer Vision and
 *              Pattern Recognition. 2015.},
 *   url     = {https://arxiv.org/abs/1409.4842},
 *   eprint  = {1409.4842}
 *   year    = {2015},
 * }
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam CustomLayers Additional custom layers if required.
 */
template <
  typename InputDataType = arma::mat,
  typename OutputDataType = arma::mat,
  typename... CustomLayers
>
class InceptionLayer
{
 public:
  //! Create the InceptionLayer object.
  InceptionLayer();

  /**
   * Create the InceptionLayer layer object.
   *
   * @param inSize The number of input maps.
   * @param inputWidth The width of the input data.
   * @param inputHeight The height of the input data.
   * @param outOne The number of output maps of the first network.
   * @param outTwo The number of output maps of the second network.
   * @param outThree The number of output maps of the third network.
   * @param outFour The number of output maps of the fourth network.
   * @param outMidTwo The number of output maps of the middle layer of the
   *                  second network.
   * @param outMidThree The number of output maps of the middle layer of the
   *                    third network.
   */
  InceptionLayer(const size_t inSize,
                 const size_t inputWidth,
                 const size_t inputHeight,
                 const size_t outOne,
                 const size_t outTwo,
                 const size_t outThree,
                 const size_t outFour,
                 const size_t outMidTwo,
                 const size_t outMidThree);

  //! Destructor to release allocated memory.
  ~InceptionLayer();

  /**
   * Forward pass of the InceptionLayer layer.
   * It calls the Forward() function of the concat layer object.
   *
   * @param input Input data for the layer.
   * @param output Resulting output activations.
   */
  template<typename eT>
  void Forward(arma::Mat<eT>&& input, arma::Mat<eT>&& output);

  /**
   * Backward pass through the layer. This function calls the Backward()
   * function of the concat layer object.
   *
   * @param input The input activations.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& input,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g);

  /**
   * Calculate the gradients. This basically will also call the gradient
   * function of the concat layer object.
   *
   * @param input The input activations.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Mat<eT>&& input,
                arma::Mat<eT>&& error,
                arma::Mat<eT>&& gradient);

  //! Get the delta.
  OutputDataType const& Delta() const { return layer->Delta(); }
  //! Modify the delta.
  OutputDataType& Delta() { return layer->Delta(); }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return layer->Gradient(); }
  //! Modify the gradient.
  OutputDataType& Gradient() { return layer->Gradient(); }

  arma::mat const& InputParameter() const { return layer->InputParameter(); }
  //! Modify the input parameter.
  arma::mat& InputParameter() { return layer->InputParameter(); }

  //! Return the model modules.
  std::vector<LayerTypes<CustomLayers...> >& Model(){ return layer->Model(); }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const
  {
    return layer->OutputParameter();
  }
  //! Modify the output parameter.
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
}; // class BatchNorm

} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "inception_layer_impl.hpp"

#endif
