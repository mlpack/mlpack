/**
 * @file methods/ann/layer/replicate.hpp
 * @author Adam Kropp
 *
 * Definition of the Replicate class, which replicates the input n times
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REPLICATE_HPP
#define MLPACK_METHODS_ANN_LAYER_REPLICATE_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Replicate class. The Replicate class replicates the
 * input n times along a specified axis.  The output will have the same number
 * of dimnensions as the input, with all dimensions other than the one
 * specified in axis being the same size as the input.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template <typename MatType = arma::mat>
class ReplicateType : public Layer<MatType>
{
 public:
  /**
   * Create the Replicate object.  The axis used for replication will be the last
   * one.
   */
  ReplicateType();

  /**
   * Create the Replicate object, specifying a particular axis on which the layer
   * outputs should be replicated.
   *
   * @param n Number of times to replicate
   * @param axis Replicate axis.
   */
  ReplicateType(const size_t n, const size_t axis = 0);

  /**
   * Destroy the layers held by the model.
   */
  virtual ~ReplicateType();

  //! Clone the ReplicateType object. This handles polymorphism correctly.
  ReplicateType* Clone() const override { return new ReplicateType(*this); }

  //! Copy the given ReplicateType layer.
  ReplicateType(const ReplicateType& other);
  //! Take ownership of the given ReplicateType layer.
  ReplicateType(ReplicateType&& other);
  //! Copy the given ReplicateType layer.
  ReplicateType& operator=(const ReplicateType& other);
  //! Take ownership of the given ReplicateType layer.
  ReplicateType& operator=(ReplicateType&& other);

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

  //! Get the axis of replicateenation.
  size_t Axis() const { return axis; }

  size_t N() const { return n; }

  void ComputeOutputDimensions() override
  {

    const size_t numOutputDimensions = this->inputDimensions.size();

    // If the user did not specify an axis, we will use the last one.
    // Otherwise, we must sanity check to ensure that the axis we are
    // replicating along is valid.
    if (!useAxis)
    {
      axis = this->inputDimensions.size() - 1;
    }
    else if (axis >= numOutputDimensions)
    {
      std::ostringstream oss;
      oss << "Replicate::ComputeOutputDimensions(): cannot replicate outputs "
          << "along axis " << axis << " when input only has "
          << this->inputDimensions.size() << " axes!";
      throw std::invalid_argument(oss.str());
    }

    // Now, we replicate the output along a specific axis.
    this->outputDimensions = this->inputDimensions;
    this->outputDimensions[axis] *= n;

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
  //! Parameter which indicates the axis of replicateenation.
  size_t axis;

  size_t n;

  //! Parameter which indicates whether to use the axis of replication.
  bool useAxis;

  size_t aliasRows;
  size_t aliasCols;
}; // class ReplicateType.

// Standard Replicate layer.
typedef ReplicateType<arma::mat> Replicate;

} // namespace mlpack

// Include implementation.
#include "replicate_impl.hpp"

#endif
