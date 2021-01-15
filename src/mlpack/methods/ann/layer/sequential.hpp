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

#include "layer.hpp"

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
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam Residual If true, use the object as a Residual block.
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat,
    bool Residual = false
>
class SequentialType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the Sequential object using the specified parameters.
   *
   * @param model Expose the all network modules.
   */
  SequentialType(const bool model = true);

  /**
   * Create the Sequential object using the specified parameters.
   *
   * @param model Expose all the network modules.
   * @param ownsLayers If true, then this module will delete its layers when
   *      deallocated.
   */
  SequentialType(const bool model, const bool ownsLayers);

  //! Copy constructor.
  SequentialType(const SequentialType& layer);

  //! Copy assignment operator.
  SequentialType& operator = (const SequentialType& layer);

  //! Destroy the Sequential object.
  ~SequentialType();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
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
                OutputType& /* gradient */);

  /**
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args) { network.push_back(new LayerType(args...)); }

  /**
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(Layer<InputType, OutputType>* layer) { network.push_back(layer); }

  //! Return the model modules.
  std::vector<Layer<InputType, OutputType>*>& Model()
  {
    if (model)
    {
      return network;
    }

    return empty;
  }

  //! Return the initial point for the optimization.
  const OutputType& Parameters() const { return parameters; }
  //! Modify the initial point for the optimization.
  OutputType& Parameters() { return parameters; }

  //! Get the input parameter.
  const InputType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  const OutputType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  const OutputType& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the gradient.
  const OutputType& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputType& Gradient() { return gradient; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Parameter which indicates if the modules should be exposed.
  bool model;

  //! Indicator if we already initialized the model.
  bool reset;

  //! Locally-stored network modules.
  std::vector<Layer<InputType, OutputType>*> network;

  //! Locally-stored model parameters.
  OutputType parameters;

  //! Locally-stored empty list of modules.
  std::vector<Layer<InputType, OutputType>*> empty;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored input parameter object.
  InputType inputParameter;

  //! Locally-stored output parameter object.
  OutputType outputParameter;

  //! Locally-stored gradient object.
  OutputType gradient;

  //! The input width.
  size_t width;

  //! The input height.
  size_t height;

  //! Whether we are responsible for deleting the layers held in this module.
  bool ownsLayers;
}; // class SequentialType

// Standard Sequential layer.
typedef SequentialType<arma::mat, arma::mat, false> Sequential;

// Standard Residual layer.
typedef SequentialType<arma::mat, arma::mat, true> Residual;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "sequential_impl.hpp"

#endif
