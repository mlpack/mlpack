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

#include "layer.hpp"

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
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class HighwayType : public Layer<InputType, OutputType>
{
 public:
  //! Create the HighwayTest object.
  HighwayType();

  /**
   * Create the HighwayTest object.
   *
   * @param inSize The number of input units.
   * @param model Expose all the network modules.
   */
  HighwayType(const size_t inSize, const bool model = true);

  //! Destroy the Highway object.
  ~HighwayType();

	//! Clone the HighwayType object. This handles polymorphism correctly.
	HighwayType* Clone() const { return new HighwayType(*this); }

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
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed-backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the
   * feed-forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

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
  void Add(Layer<arma::mat, arma::mat>* layer)
  {
    network.push_back(layer);
    networkOwnerships.push_back(false);
  }

  //! Return the modules of the model.
  std::vector<Layer<>*>& Model()
  {
    if (model)
    {
      return network;
    }

    return empty;
  }

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the input parameter.
  InputType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the gradient.
  OutputType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputType& Gradient() { return gradient; }

  //! Get the number of input units.
  size_t InSize() const { return inSize; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Parameter which indicates if the modules should be exposed.
  bool model;

  //! Indicator if we already initialized the model.
  bool reset;

  //! Locally-stored network modules.
  std::vector<Layer<arma::mat, arma::mat>*> network;

  //! The list of network modules we are responsible for.
  std::vector<bool> networkOwnerships;

  //! Locally-stored empty list of modules.
  std::vector<Layer<arma::mat, arma::mat>*> empty;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored gradient object.
  OutputType gradient;

  //! Weights for transformation of output.
  OutputType transformWeight;

  //! Bias for transformation of output.
  OutputType transformBias;

  //! Locally-stored transform gate parameters.
  OutputType transformGate;

  //! Locally-stored transform gate activation.
  OutputType transformGateActivation;

  //! Locally-stored transform gate error.
  OutputType transformGateError;

  //! Locally-stored input parameter object.
  InputType inputParameter;

  //! Locally-stored output parameter object.
  OutputType outputParameter;

  //! The input width.
  size_t width;

  //! The input height.
  size_t height;

  //! The normal output without highway network.
  OutputType networkOutput;
}; // class HighwayType

// Standard Highway layer.
typedef HighwayType<arma::mat, arma::mat> Highway;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "highway_impl.hpp"

#endif
