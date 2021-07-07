/**
 * @file methods/ann/layer/layer.hpp
 * @author Marcus Edel
 *
 * Base class for neural network layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYER_HPP

namespace mlpack {
namespace ann {

/**
 * A layer is an abstract class implementing common neural networks operations,
 * such as convolution, batch norm, etc. These operations require managing
 * weights, losses, updates, and inter-layer connectivity.
 *
 * Users will just instantiate a layer by inherited from the abstract class and
 * implement the layer specific methods. It is recommend that descendants of
 * Layer implement the following methods:
 *
 *  - Constructor: Defines custom layer attributes, and creates layer state
 *    variables.
 *
 *  - Forward(input, output): Performs the forward logic of applying the layer
 *    to the input object and storing the result in the output object.
 *
 *  - Backward(input, gy, g): Performs a backpropagation step through the layer,
 *    with respect to the given input.
 *
 *  - Gradient(input, error, gradient): Computing the gradient of the layer with
 *    respect to its own input.
 *
 * The memory for the layer's parameters (weights and biases) is not allocated
 * by the layer itself, instead it is allocated by the network that the layer
 * belongs to, and passed to the layer when it needs to use it.
 *
 * See the linear layer implementation for a basic example. It's a layer with
 * two variables, w and b, that returns y = w * x + b. It shows how to implement
 * Forward(), Backward() and Gradient().  The weights of the layers are tracked
 * in layer.Parameters().
 *
 * @tparam InputType The type of the layer's inputs. Layers automatically cast
 *     inputs to this type (default: arma::mat).
 * @tparam OutputType The type of the layer's computation which also causes the
 *     computations and output to also be in this type. The type also allows the
 *     computation and weight type to differ from the input type
 *     (default: arma::mat).
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class Layer
{
 public:
  //! Default constructor.
  Layer() : validOutputDimensions(false) { /* Nothing to do here */ }

  //! Default deconstructor.
  virtual ~Layer() { }

  //! Copy constructor.
  Layer(const Layer& /* layer */) { /* Nothing to do here */ }

  //! Make a copy of the object.
  virtual Layer* Clone() const = 0;

  //! Move constructor.
  Layer(Layer&& /* layer */) { /* Nothing to do here */ }

  //! Copy assignment operator.
  virtual Layer& operator=(const Layer& /* layer */) { return *this; }

  //! Move assignment operator.
  virtual Layer& operator=(Layer&& /* layer */) { return *this; }

  /**
   * Takes an input object, and computes the corresponding output of the layer.
   * In general input and output are matrices. However, some special layers like
   * table layers might expect something else. Please, refer to each layer
   * specification for further information.
   *
   * @param * (input) Input data used for evaluating the specified layer.
   * @param * (output) Resulting output.
   */
  virtual void Forward(const InputType& /* input */,
                       OutputType& /* output */)
  { /* Nothing to do here */ }

  /**
   * Takes an input and output object, and computes the corresponding loss of
   * the layer.  In general input and output are matrices. However, some special
   * layers like table layers might expect something else. Please, refer to each
   * layer specification for further information.
   *
   * @param * (input) Input data used for evaluating the specified layer.
   * @param * (output) Resulting output.
   */
  virtual void Forward(const InputType& /* input */,
                       const OutputType& /* output */)
  { /* Nothing to do here */ }

  /**
   * Performs a backpropagation step through the layer, with respect to the
   * given input. In general this method makes the assumption Forward(input,
   * output) has been called before, with the same input. If you do not respect
   * this rule, Backward(input, gy, g) might compute incorrect results.
   *
   * In general input and gy and g are matrices. However, some special
   * sub-classes like table layers might expect something else. Please, refer to
   * each module specification for further information.
   *
   * A backpropagation step consist of computing of computing the gradient
   * output input with respect to the output of the layer and given error.
   *
   * During the backward pass our goal is to use 'gy' in order to compute the
   * downstream gradients (g). We assume that the upstream gradient (gy) has
   * already been computed and is passed to the layer.
   *
   * @param * (input) The propagated input activation.
   * @param * (gy) The backpropagated error.
   * @param * (g) The calculated gradient.
   */
  virtual void Backward(const InputType& /* input */,
                        const OutputType& /* gy */,
                        OutputType& /* g */)
  { /* Nothing to do here */ }

  /**
   * Computing the gradient of the layer with respect to its own input. This is
   * returned in gradient.
   *
   * The layer parameters (weights and biases) are updated accordingly using the
   * computed gradient not by the layer itself, instead they are updated by the
   * network that holds the instantiated layer.
   *
   * @param * (input) The input parameter used for calculating the gradient.
   * @param * (error) The calculated error.
   * @param * (gradient) The calculated gradient.
   */
  virtual void Gradient(const InputType& /* input */,
                        const OutputType& /* error */,
                        OutputType& /* gradient */)
  { /* Nothing to do here */ }

  /**
   * Reset the layer parameter. The method is called to assigned the allocated
   * memory to the internal layer parameters like weights and biases. The method
   * should be called before the first call of Forward(input, output). If you
   * do not respect this rule, Forward(input, output) and Backward(input, gy, g)
   * might compute incorrect results.
   */
  virtual void SetWeights(typename OutputType::elem_type* /* weightsPtr */) { }

  virtual size_t WeightSize() const { return 0; }

  // TODO: figure out what to do
  /**
   * Resets the cell to accept a new input. This breaks the BPTT chain starts a
   * new one.
   *
   * @param * (size) The current maximum number of steps through time.
   */
  virtual void ResetCell(const size_t /* size */) {}

  /**
   * Get whether the layer is currently in training mode.
   *
   * @note During network training, this should be set to `true` for each layer
   * in the network, and when predicting/testing the network, this should be set
   * to `false`.  (This is handled automatically by the `FFN` class and other
   * related classes.)
   */
  virtual bool const& Training() const { return training; }

  /**
   * Modify whether the layer is currently in training mode.
   *
   * @note During network training, this should be set to `true` for each layer
   * in the network, and when predicting/testing the network, this should be set
   * to `false`.  (This is handled automatically by the `FFN` class and other
   * related classes.)
   */
  virtual bool& Training() { return training; }

  //! Get the layer loss.  Overload this if the layer should add any extra loss
  //! to the loss function when computing the objective.  (TODO: better comment)
  virtual double Loss() { return 0; }

  // TODO: these need better comments.

  //! Get the input dimensions.
  const std::vector<size_t>& InputDimensions() const { return inputDimensions; }
  //! Modify the input dimensions.
  std::vector<size_t>& InputDimensions()
  {
    validOutputDimensions = false;
    return inputDimensions;
  }

  //! Get the output dimensions.
  const std::vector<size_t>& OutputDimensions()
  {
    if (!validOutputDimensions)
    {
      this->ComputeOutputDimensions();
      validOutputDimensions = true;
    }

    return outputDimensions;
  }

  //! Get the parameters.
  virtual const OutputType& Parameters() const { return OutputType(); }
  //! Set the parameters.
  virtual OutputType& Parameters()
  {
    throw std::invalid_argument("Layer::Parameters(): cannot modify parameters "
        "of a layer with no weights!");
  }

  //! Compute the output dimensions.  This should be overloaded if the layer is
  //! meant to work on higher-dimensional objects.  When this is called, it is a
  //! safe assumption that InputDimensions() is correct.
  virtual void ComputeOutputDimensions()
  {
    // The default implementation is to assume that the output size is the same
    // as the input.
    outputDimensions = inputDimensions;
  }

  //! Get the number of elements in the output from this layer.
  //! This is marked final because no class should ever need to override
  //! this---instead, override OutputDimensions()!
  // TODO: is final not available in C++11?
  size_t OutputSize()
  {
    if (!validOutputDimensions)
    {
      this->ComputeOutputDimensions();
      validOutputDimensions = true;
    }

    return std::accumulate(this->outputDimensions.begin(),
        this->outputDimensions.end(), 0);
  }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(inputDimensions);
    ar(outputDimensions);
    ar(training);

    // Note that layer weights are serialized by the FFN!
  }

 protected:
  // This matrix holds the parameters of the 

  // TODO: comment
  std::vector<size_t> inputDimensions;
  std::vector<size_t> outputDimensions;
  bool validOutputDimensions;

  //! If true, the layer is in training mode; otherwise, it is in testing mode.
  bool training;
};

} // namespace ann
} // namespace mlpack

#endif
