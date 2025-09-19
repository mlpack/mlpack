/**
 * @file methods/ann/layer/sum_reduce.hpp
 * @author Andrew Furey
 *
 * Definition of the SumReduce class sums inputs along a given axis.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_REDUCE_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_REDUCE_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the SumReduce layer. The SumReduce module sums its
 * incoming data along a given axis.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class SumReduce : public Layer<MatType>
{
 public:
  /**
   * Create the SumReduce object.
   */
  SumReduce(size_t axis = 0, bool keepDimensions = false);

  //! Clone the SumReduce object. This handles polymorphism correctly.
  SumReduce* Clone() const { return new SumReduce(*this); }

  // Virtual destructor.
  virtual ~SumReduce() { }

  //! Copy the given SumReduce layer.
  SumReduce(const SumReduce& other);
  //! Take ownership of the given SumReduce layer.
  SumReduce(SumReduce&& other);
  //! Copy the given SumReduce layer.
  SumReduce& operator=(const SumReduce& other);
  //! Take ownership of the given SumReduce layer.
  SumReduce& operator=(SumReduce&& other);

  using CubeType = typename GetCubeType<MatType>::type;

  /**
   * Forward pass: Sum along the given axis.
   *
   * @param input Input data
   * @param output Resulting summed output.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Backward pass: send delta backwards (expand along the given axis)
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& output,
                const MatType& gy,
                MatType& g);

  //! Compute the output dimensions of the layer, based on the internal values
  //! of `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored axis to reduce.
  size_t axis;

  //! If true, the dimension that gets summed is set to have size 1, otherwise
  //! it is deleted and the output will have one fewer dimension than the input.
  bool keepDimensions;

  //! Number of rows in cube alias.
  size_t rows;
  //! Number of slices in cube alias.
  size_t slices;
}; // class SumReduce

} // namespace mlpack

// Include implementation.
#include "sum_reduce_impl.hpp"

#endif
