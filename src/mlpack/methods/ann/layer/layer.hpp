/**
 * @file methods/ann/layer/layer.hpp
 * @author Marcus Edel
 *
 * Base class for neural network layers, and convenience include for all layer
 * types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYER_HPP

namespace mlpack {

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
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class Layer
{
 public:
  //! Default constructor.
  Layer() :
      validOutputDimensions(false),
      training(false)
  { /* Nothing to do here */ }

  //! Default deconstructor.
  virtual ~Layer() { /* Nothing to do here */ }

  //! Copy constructor.  This is not responsible for copying weights!
  Layer(const Layer& layer) :
      inputDimensions(layer.inputDimensions),
      outputDimensions(layer.outputDimensions),
      validOutputDimensions(layer.validOutputDimensions),
      training(layer.training)
  { /* Nothing to do here */ }

  //! Make a copy of the object.
  virtual Layer* Clone() const = 0;

  //! Move constructor.  This is not responsible for moving weights!
  Layer(Layer&& layer) :
      inputDimensions(std::move(layer.inputDimensions)),
      outputDimensions(std::move(layer.outputDimensions)),
      validOutputDimensions(std::move(layer.validOutputDimensions)),
      training(std::move(layer.training))
  { /* Nothing to do here */ }

  //! Copy assignment operator.  This is not responsible for copying weights!
  Layer& operator=(const Layer& layer)
  {
    if (&layer != this)
    {
      inputDimensions = layer.inputDimensions;
      outputDimensions = layer.outputDimensions;
      validOutputDimensions = layer.validOutputDimensions;
      training = layer.training;
    }

    return *this;
  }

  //! Move assignment operator.  This is not responsible for moving weights!
  Layer& operator=(Layer&& layer)
  {
    if (&layer != this)
    {
      inputDimensions = std::move(layer.inputDimensions);
      outputDimensions = std::move(layer.outputDimensions);
      validOutputDimensions = std::move(layer.validOutputDimensions);
      training = std::move(layer.training);
    }

    return *this;
  }

  /**
   * Takes an input object, and computes the corresponding output of the layer.
   * In general input and output are matrices. However, some special layers like
   * table layers might expect something else. Please, refer to each layer
   * specification for further information.
   *
   * @param * (input) Input data used for evaluating the specified layer.
   * @param * (output) Resulting output.
   */
  virtual void Forward(const MatType& /* input */,
                       MatType& /* output */)
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
  virtual void Forward(const MatType& /* input */,
                       const MatType& /* output */)
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
   * @param * (input) The input data (x) given to the forward pass.
   * @param * (output) The propagated data (f(x)) resulting from Forward()
   * @param * (gy) The backpropagated error.
   * @param * (g) The calculated gradient.
   */
  virtual void Backward(const MatType& /* input */,
                        const MatType& /* output */,
                        const MatType& /* gy */,
                        MatType& /* g */)
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
  virtual void Gradient(const MatType& /* input */,
                        const MatType& /* error */,
                        MatType& /* gradient */)
  { /* Nothing to do here */ }

  /**
   * Reset the layer parameter. The method is called to assigned the allocated
   * memory to the internal layer parameters like weights and biases. The method
   * should be called before the first call of Forward(input, output). If you
   * do not respect this rule, Forward(input, output) and Backward(input, gy, g)
   * might compute incorrect results.
   *
   * @param weightsPtr This pointer should be used as the first element of the
   *    memory that is allocated for this layer.  In general, SetWeights()
   *    implementations should use MakeAlias() with weightsPtr to wrap the
   *    weights of a layer.
   */
  virtual void SetWeights(const MatType& /* weightsIn */) { }

  /**
   * Get the total number of trainable weights in the layer.
   */
  virtual size_t WeightSize() const { return 0; }

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
  virtual double Loss() const { return 0; }

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
  virtual const MatType& Parameters() const
  {
    throw std::invalid_argument("Layer::Parameters(): cannot access parameters "
        "of a layer with no weights!");
  }
  //! Set the parameters.
  virtual MatType& Parameters()
  {
    throw std::invalid_argument("Layer::Parameters(): cannot modify parameters "
        "of a layer with no weights!");
  }

  /**
   * Override the weight matrix of the layer. This method is used to set the
   * weights of the layer, and is only used during layer weight initialization.
   * This method should be used if you want initialize the weights of the layer
   * with a custom matrix. We will provide the weight matrix on which you can
   * override the weights of the layer.
   *
   * @param * (W) Weight matrix to initialize.
   * @param * (elements) Number of elements.
   */
  virtual void CustomInitialize(
      MatType& /* W */,
      const size_t /* elements */)
  { /* Nothing to do here */ }

  //! Compute the output dimensions.  This should be overloaded if the layer is
  //! meant to work on higher-dimensional objects.  When this is called, it is a
  //! safe assumption that InputDimensions() is correct.
  virtual void ComputeOutputDimensions()
  {
    // The default implementation is to assume that the output size is the same
    // as the input.
    outputDimensions = inputDimensions;
  }

  //! Get the number of elements in the output from this layer.  This cannot be
  //! overloaded!  Overload `ComputeOutputDimensions()` instead.
  virtual size_t OutputSize() final
  {
    if (!validOutputDimensions)
    {
      this->ComputeOutputDimensions();
      validOutputDimensions = true;
    }

    size_t outputSize = 1;
    for (size_t i = 0; i < this->outputDimensions.size(); ++i)
      outputSize *= this->outputDimensions[i];
    return outputSize;
  }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(inputDimensions));
    ar(CEREAL_NVP(outputDimensions));
    ar(CEREAL_NVP(validOutputDimensions));
    ar(CEREAL_NVP(training));

    // Note that layer weights are serialized by the FFN!
  }

 protected:
  /**
   * Logical input dimensions of each point.  Although each point given to !
   * `Forward()` will be represented as a column in a matrix, logically
   * speaking it can be a higher-order tensor.  So, for instance, if the point
   * is 2-dimensional images of size 10x10, `Forward()` will contain columns
   * with 100 rows, and `inputDimensions` will be `{10, 10}`.  This generalizes
   * to higher dimensions.
   */
  std::vector<size_t> inputDimensions;

  /**
   * Logical output dimensions of each point.  If the layer only performs
   * elementwise operations, this is most likely equal to `inputDimensions`; but
   * if the layer performs more complicated transformations, it may be
   * different.
   */
  std::vector<size_t> outputDimensions;

  //! This is `true` if `ComputeOutputDimensions()` has been called, and
  //! `outputDimensions` can be considered to be up-to-date.
  bool validOutputDimensions;

  //! If true, the layer is in training mode; otherwise, it is in testing mode.
  bool training;
};

} // namespace mlpack

// Now include all of the layer types.
#include "layer_types.hpp"

#endif
