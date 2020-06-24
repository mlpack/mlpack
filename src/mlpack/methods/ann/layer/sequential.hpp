/**
 * @file methods/ann/layer/sequential.hpp
 * @author Marcus Edel
 *
 * Definition of the Sequential class, which acts as a feed-forward fully
 * connected network container.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_HPP
#define MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_HPP

#include <mlpack/prereqs.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include "../visitor/delete_visitor.hpp"
#include "../visitor/copy_visitor.hpp"
#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_height_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"
#include "../visitor/output_width_visitor.hpp"

#include "layer_types.hpp"
#include "add_merge.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Sequential class. The sequential class works as a
 * feed-forward fully connected network container which plugs various layers
 * together.
 *
 * This class can also be used as a container for a residual block. In that
 * case, the sizes of the input and output matrices of this class should be
 * equal. A typedef has been added for use as a Residual<> class.
 *
 * For more information, refer the following paper.
 *
 * @code
 * @article{He15,
 *   author    = {Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun},
 *   title     = {Deep Residual Learning for Image Recognition},
 *   year      = {2015},
 *   url       = {https://arxiv.org/abs/1512.03385},
 *   eprint    = {1512.03385},
 * }
 * @endcode
 *
 * Note: If this class is used as the first layer of a network, it should be
 *       preceded by IdentityLayer<>.
 *
 * Note: This class should at least have two layers for a call to its Gradient()
 *       function.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam Residual If true, use the object as a Residual block.
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    bool Residual = false,
    typename... CustomLayers
>
class Sequential
{
 public:
  /**
   * Create the Sequential object using the specified parameters.
   *
   * @param model Expose the all network modules.
   */
  Sequential(const bool model = true);

  /**
   * Create the Sequential object using the specified parameters.
   *
   * @param model Expose all the network modules.
   * @param ownsLayers If true, then this module will delete its layers when
   *      deallocated.
   */
  Sequential(const bool model, const bool ownsLayers);

  //! Copy constructor.
  Sequential(const Sequential& layer);

  //! Copy assignment operator.
  Sequential& operator = (const Sequential& layer);

  //! Destroy the Sequential object.
  ~Sequential();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& /* gradient */);

  /*
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args) { network.push_back(new LayerType(args...)); }

  /*
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(LayerTypes<CustomLayers...> layer) { network.push_back(layer); }

  //! Return the model modules.
  std::vector<LayerTypes<CustomLayers...> >& Model()
  {
    if (model)
    {
      return network;
    }

    return empty;
  }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameters; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameters; }

  //! Get the input parameter.
  arma::mat const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  arma::mat& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  arma::mat const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  arma::mat& OutputParameter() { return outputParameter; }

  //! Get the delta.
  arma::mat const& Delta() const { return delta; }
  //! Modify the delta.
  arma::mat& Delta() { return delta; }

  //! Get the gradient.
  arma::mat const& Gradient() const { return gradient; }
  //! Modify the gradient.
  arma::mat& Gradient() { return gradient; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */);

 private:
  //! Parameter which indicates if the modules should be exposed.
  bool model;

  //! Indicator if we already initialized the model.
  bool reset;

  //! Locally-stored network modules.
  std::vector<LayerTypes<CustomLayers...> > network;

  //! Locally-stored model parameters.
  arma::mat parameters;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! Locally-stored empty list of modules.
  std::vector<LayerTypes<CustomLayers...> > empty;

  //! Locally-stored delta object.
  arma::mat delta;

  //! Locally-stored input parameter object.
  arma::mat inputParameter;

  //! Locally-stored output parameter object.
  arma::mat outputParameter;

  //! Locally-stored gradient object.
  arma::mat gradient;

  //! Locally-stored output width visitor.
  OutputWidthVisitor outputWidthVisitor;

  //! Locally-stored output height visitor.
  OutputHeightVisitor outputHeightVisitor;

  //! Locally-stored copy visitor
  CopyVisitor<CustomLayers...> copyVisitor;

  //! The input width.
  size_t width;

  //! The input height.
  size_t height;

  //! Whether we are responsible for deleting the layers held in this module.
  bool ownsLayers;
}; // class Sequential

/*
 * Convenience typedef for use as Residual<> layer.
 */
template<
  typename InputDataType = arma::mat,
  typename OutputDataType = arma::mat,
  typename... CustomLayers
>
using Residual = Sequential<
    InputDataType, OutputDataType, true, CustomLayers...>;

} // namespace ann
} // namespace mlpack

//! Set the serialization version of the Sequential class.
namespace boost {
namespace serialization {

template <
    typename InputDataType,
    typename OutputDataType,
    bool Residual,
    typename... CustomLayers
>
struct version<mlpack::ann::Sequential<
    InputDataType, OutputDataType, Residual, CustomLayers...>>
{
  BOOST_STATIC_CONSTANT(int, value = 1);
};

} // namespace serialization
} // namespace boost

// Include implementation.
#include "sequential_impl.hpp"

#endif
