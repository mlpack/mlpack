/**
 * @file methods/ann/layer/highway.hpp
 * @author Konstantin Sidorov
 * @author Saksham Bansal
 *
 * Definition of the Highway layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HIGHWAY_HPP
#define MLPACK_METHODS_ANN_LAYER_HIGHWAY_HPP

#include <mlpack/prereqs.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include "../visitor/delete_visitor.hpp"
#include "../visitor/delta_visitor.hpp"
#include "../visitor/output_height_visitor.hpp"
#include "../visitor/output_parameter_visitor.hpp"
#include "../visitor/output_width_visitor.hpp"

#include "layer_types.hpp"
#include "add_merge.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Highway layer. The Highway class can vary its behavior
 * between that of feed-forward fully connected network container and that
 * of a layer which simply passes its inputs through depending on the transform
 * gate. Note that the size of the input and output matrices of this class
 * should be equal.
 *
 * For more information, refer the following paper.
 *
 * @code
 * @article{Srivastava2015,
 *   author  = {Rupesh Kumar Srivastava, Klaus Greff, Jurgen Schmidhuber},
 *   title   = {Training Very Deep Networks},
 *   journal = {Advances in Neural Information Processing Systems},
 *   year    = {2015},
 *   url     = {https://arxiv.org/abs/1507.06228},
 * }
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename... CustomLayers>
class Highway
{
 public:
  //! Create the Highway object.
  Highway();

  /**
   * Create the Highway object.
   *
   * @param inSize The number of input units.
   * @param model Expose all the network modules.
   */
  Highway(const size_t inSize, const bool model = true);

  //! Destroy the Highway object.
  ~Highway();

  /**
   * Reset the layer parameter.
   */
  void Reset();

  /**
   * Ordinary feed-forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed-backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the
   * feed-forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& gradient);

  /**
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args)
  {
    network.push_back(new LayerType(args...));
    networkOwnerships.push_back(true);
  }

  /**
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(LayerTypes<CustomLayers...> layer)
  {
    network.push_back(layer);
    networkOwnerships.push_back(false);
  }

  //! Return the modules of the model.
  std::vector<LayerTypes<CustomLayers...> >& Model()
  {
    if (model)
    {
      return network;
    }

    return empty;
  }

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

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

  //! Get the number of input units.
  size_t InSize() const { return inSize; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Parameter which indicates if the modules should be exposed.
  bool model;

  //! Indicator if we already initialized the model.
  bool reset;

  //! Locally-stored network modules.
  std::vector<LayerTypes<CustomLayers...> > network;

  //! The list of network modules we are responsible for.
  std::vector<bool> networkOwnerships;

  //! Locally-stored empty list of modules.
  std::vector<LayerTypes<CustomLayers...> > empty;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Weights for transformation of output.
  OutputDataType transformWeight;

  //! Bias for transformation of output.
  OutputDataType transformBias;

  //! Locally-stored transform gate parameters.
  OutputDataType transformGate;

  //! Locally-stored transform gate activation.
  OutputDataType transformGateActivation;

  //! Locally-stored transform gate error.
  OutputDataType transformGateError;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The input width.
  size_t width;

  //! The input height.
  size_t height;

  //! The normal output without highway network.
  OutputDataType networkOutput;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! Locally-stored output width visitor.
  OutputWidthVisitor outputWidthVisitor;

  //! Locally-stored output height visitor.
  OutputHeightVisitor outputHeightVisitor;
}; // class Highway

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "highway_impl.hpp"

#endif
