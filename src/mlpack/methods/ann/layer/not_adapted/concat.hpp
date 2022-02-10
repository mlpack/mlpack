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
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Concat class. The Concat class works as a
 * feed-forward fully connected network container which plugs various layers
 * together.
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
class ConcatType : public MultiLayer<InputType, OutputType>
{
 public:
  /**
   * Create the Concat object using the specified parameters.
   *
   * @param run Call the Forward/Backward method before the output is merged.
   */
  ConcatType(const bool run = true);

  /**
   * Create the Concat object, specifying a particular axis on which the layer
   * outputs should be concatenated.
   *
   * @param axis Concat axis.
   * @param run Call the Forward/Backward method before the output is merged.
   */
  ConcatType(const size_t axis, const bool run = true);

  /**
   * Destroy the layers held by the model.
   */
  ~ConcatType();

  //! Clone the ConcatType object. This handles polymorphism correctly.
  ConcatType* Clone() const { return new ConcatType(*this); }

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
   * This is the overload of Backward() that runs only a specific layer with
   * the given input.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   * @param index The index of the layer to run.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g,
                const size_t index);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& /* input */,
                const OutputType& error,
                OutputType& /* gradient */);

  /**
   * This is the overload of Gradient() that runs a specific layer with the
   * given input.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   * @param The index of the layer to run.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient,
                const size_t index);

  //! Get the value of run parameter.
  bool Run() const { return run; }
  //! Modify the value of run parameter.
  bool& Run() { return run; }

  //! Get the axis of concatenation.
  const size_t& ConcatAxis() const { return axis; }

  //! Get the size of the weight matrix.
  size_t WeightSize() const { return 0; }

  void ComputeOutputDimensions()
  {
    // The input is sent to every layer.
    for (size_t i = 0; i < network.size(); ++i)
    {
      network[i]->InputDimensions() = this->inputDimensions;
      network[i]->ComputeOutputDimensions();
    }

    // If the user did not specify an axis, we will use the last one.
    // Otherwise, we must sanity check to ensure that the axis we are
    // concatenating along is valid.
    if (!useAxis)
    {
      axis  = this->inputDimensions.size() - 1;
    }
    else if (axis >= this->inputDimensions.size())
    {
      std::ostringstream oss;
      oss << "Concat::ComputeOutputDimensions(): cannot concatenate outputs "
          << "along axis " << axis << " when input only has "
          << this->inputDimensions.size() << " axes!";
      throw std::invalid_argument(oss.str());
    }

    // Now, we concatenate the output along a specific axis.
    this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
        0);
    for (size_t i = 0; i < this->inputDimensions.size(); ++i)
    {
      if (i == axis)
      {
        // Accumulate output size along this axis for each layer output.
        for (size_t n = 0; n < this->network.size(); ++n)
        {
          this->outputDimensions[i] += this->network[n]->OutputDimensions()[i];
        }
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
  }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar,  const uint32_t /* version */);

 private:
  //! Parameter which indicates the axis of concatenation.
  size_t axis;

  //! Parameter which indicates whether to use the axis of concatenation.
  bool useAxis;
}; // class ConcatType.

// Standard Concat layer.
typedef ConcatType<arma::mat, arma::mat> Concat;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concat_impl.hpp"

#endif
