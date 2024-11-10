/**
 * @file methods/ann/layer/concat.hpp
 * @author Marcus Edel
 * @author Mehul Kumar Nirala
 *
 * Definition of the Concat class, which acts as a concatenation container.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Concat class. The Concat class works as a
 * feed-forward fully connected network container which plugs various layers
 * together.
 *
 * NOTE: this class is not intended to exist for long!  It will be replaced with
 * a more flexible DAG network type.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template <typename MatType = arma::mat>
class ConcatType : public MultiLayer<MatType>
{
 public:
  /**
   * Create the Concat object.  The axis used for concatenation will be the last
   * one.
   */
  ConcatType();

  /**
   * Create the Concat object, specifying a particular axis on which the layer
   * outputs should be concatenated.
   *
   * @param axis Concat axis.
   */
  ConcatType(const size_t axis);

  /**
   * Destroy the layers held by the model.
   */
  virtual ~ConcatType();

  //! Clone the ConcatType object. This handles polymorphism correctly.
  ConcatType* Clone() const { return new ConcatType(*this); }

  //! Copy the given ConcatType layer.
  ConcatType(const ConcatType& other);
  //! Take ownership of the given ConcatType layer.
  ConcatType(ConcatType&& other);
  //! Copy the given ConcatType layer.
  ConcatType& operator=(const ConcatType& other);
  //! Take ownership of the given ConcatType layer.
  ConcatType& operator=(ConcatType&& other);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  /**
   * This is the overload of Backward() that runs only a specific layer with
   * the given input.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   * @param index The index of the layer to run.
   */
  void Backward(const MatType& input,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g,
                const size_t index);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& /* input */,
                const MatType& error,
                MatType& /* gradient */);

  /**
   * This is the overload of Gradient() that runs a specific layer with the
   * given input.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   * @param The index of the layer to run.
   */
  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient,
                const size_t index);

  //! Get the axis of concatenation.
  size_t Axis() const { return axis; }

  // We don't need to overload WeightSize(); MultiLayer already computes this
  // correctly.  (It is the sum of weights of all child layers.)

  void ComputeOutputDimensions()
  {
    // The input is sent to every layer.
    for (size_t i = 0; i < this->network.size(); ++i)
    {
      this->network[i]->InputDimensions() = this->inputDimensions;
      this->network[i]->ComputeOutputDimensions();
    }

    const size_t numOutputDimensions = (this->network.size() == 0) ?
        this->inputDimensions.size() :
        this->network[0]->OutputDimensions().size();

    // If the user did not specify an axis, we will use the last one.
    // Otherwise, we must sanity check to ensure that the axis we are
    // concatenating along is valid.
    if (!useAxis)
    {
      axis = this->inputDimensions.size() - 1;
    }
    else if (axis >= numOutputDimensions)
    {
      std::ostringstream oss;
      oss << "Concat::ComputeOutputDimensions(): cannot concatenate outputs "
          << "along axis " << axis << " when input only has "
          << this->inputDimensions.size() << " axes!";
      throw std::invalid_argument(oss.str());
    }

    // Now, we concatenate the output along a specific axis.
    this->outputDimensions = std::vector<size_t>(numOutputDimensions, 0);
    for (size_t i = 0; i < this->outputDimensions.size(); ++i)
    {
      if (i == axis)
      {
        // Accumulate output size along this axis for each layer output.
        for (size_t n = 0; n < this->network.size(); ++n)
          this->outputDimensions[i] += this->network[n]->OutputDimensions()[i];
      }
      else
      {
        // Ensure that the output size is the same along this axis.
        const size_t axisDim = this->network[0]->OutputDimensions()[i];
        for (size_t n = 1; n < this->network.size(); ++n)
        {
          const size_t axisDim2 = this->network[n]->OutputDimensions()[i];
          if (axisDim != axisDim2)
          {
            std::ostringstream oss;
            oss << "Concat::ComputeOutputDimensions(): cannot concatenate "
                << "outputs along axis " << axis << "; held layer " << n
                << " has output size " << axisDim2 << " along axis " << i
                << ", but the first held layer has output size " << axisDim
                << "!  All layers must have identical output size in any "
                << "axis other than the concatenated axis.";
            throw std::invalid_argument(oss.str());
          }
        }

        this->outputDimensions[i] = axisDim;
      }
    }

    // Recompute total input and output sizes.  Note that we pass the input to
    // each layer held in the network, so the "total" input size (which is used
    // by the backwards pass to compute how much memory to use for holding
    // deltas) should be the number of layers multiplied by the input size for
    // each layer.
    this->totalInputSize = 1;
    this->totalOutputSize = 1;
    for (size_t i = 0; i < this->inputDimensions.size(); ++i)
      this->totalInputSize *= this->inputDimensions[i];
    this->totalInputSize *= this->network.size();
    for (size_t i = 0; i < this->outputDimensions.size(); ++i)
      this->totalOutputSize *= this->outputDimensions[i];
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Parameter which indicates the axis of concatenation.
  size_t axis;

  //! Parameter which indicates whether to use the axis of concatenation.
  bool useAxis;
}; // class ConcatType.

// Standard Concat layer.
using Concat = ConcatType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "concat_impl.hpp"

#endif
