/**
 * @file methods/ann/layer/repeat.hpp
 * @author Adam Kropp
 *
 * Definition of the Repeat class, which repeats the input n times
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REPEAT_HPP
#define MLPACK_METHODS_ANN_LAYER_REPEAT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Repeat class. The Repeat class repeats the
 * input n times along a specified axis.  The output will have the same number
 * of dimnensions as the input, with all dimensions other than the one
 * specified in axis being the same size as the input.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template <typename MatType = arma::mat>
class RepeatType : public Layer<MatType>
{
 public:
  /**
   * Create the Repeat object.  Axis defaults to 0, and n defaults to 1, meaning
   * this is the same as an Identity layer.
   */
  RepeatType();

  /**
   * Create the Repeat object, specifying a particular axis on which the layer
   * outputs should be repeatd.
   *
   * @param n Number of times to repeat
   * @param axis Repeat axis.
   */
  RepeatType(const size_t n, const size_t axis = 0);

  /**
   * Destroy the layers held by the model.
   */
  virtual ~RepeatType();

  //! Clone the RepeatType object. This handles polymorphism correctly.
  RepeatType* Clone() const override { return new RepeatType(*this); }

  //! Copy the given RepeatType layer.
  RepeatType(const RepeatType& other);
  //! Take ownership of the given RepeatType layer.
  RepeatType(RepeatType&& other);
  //! Copy the given RepeatType layer.
  RepeatType& operator=(const RepeatType& other);
  //! Take ownership of the given RepeatType layer.
  RepeatType& operator=(RepeatType&& other);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output) override;

  /**
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param * (output) The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g) override;

  //! Get the axis of repeatenation.
  size_t Axis() const { return axis; }

  size_t N() const { return n; }

  void ComputeOutputDimensions() override
  {

    const size_t numOutputDimensions = this->inputDimensions.size();

    // Sanity check to ensure that the axis we are
    // replicating along is valid.
    if (axis >= numOutputDimensions)
    {
      std::ostringstream oss;
      oss << "Repeat::ComputeOutputDimensions(): cannot repeat outputs "
          << "along axis " << axis << " when input only has "
          << this->inputDimensions.size() << " axes!";
      throw std::invalid_argument(oss.str());
    }

    // Now, we repeat the output along a specific axis.
    this->outputDimensions = this->inputDimensions;
    this->outputDimensions[axis] *= n;

    // if axis is 0, we just want the first dimension in rows, and will use
    // repelem.  If axis is > 0, we want the first axis dimensions in rows,
    // and will use repmat.
    aliasRows = this->inputDimensions[0];
    aliasCols = 1;
    for (size_t i=1; i<this->inputDimensions.size(); i++) {
      if (i < axis) {
        aliasRows *= this->inputDimensions[i];
      }
      else {
        aliasCols *= this->inputDimensions[i];
      }
    }

  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Parameter which indicates the axis of repeatenation.
  size_t axis;

  size_t n;

  size_t aliasRows;
  size_t aliasCols;
}; // class RepeatType.

// Standard Repeat layer.
typedef RepeatType<arma::mat> Repeat;

} // namespace mlpack

// Include implementation.
#include "repeat_impl.hpp"

#endif
