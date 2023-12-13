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
 * input a specified number of times along each axis.  The output will have the
 * same number of dimensions as the input, with each dimension multiplied by
 * a specified multiple.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template <typename MatType = arma::mat>
class ReplicateType : public Layer<MatType>
{
 public:
  /**
   * Create the Replicate object with no multiples.  If not set later, this
   * is equivalent to an identity layer.
   */
  ReplicateType();

  /**
   * Create the Replicate object, specifying the number of repeated blocks
   * along each dimension.
   *
   * @param multiples The number of times to repeat along each axis. Must be
   *        the same size or smaller than InputDimensions.
   */
  ReplicateType(std::vector<size_t> multiples);

  /**
   * Destroy the layers held by the model.
   */
  virtual ~ReplicateType() { }

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
   * @param * (input) The input data (x) given to the forward pass.
   * @param * (output) The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g) override;

  //! Get the replication multiples
  const std::vector<size_t>& Multiples() const { return multiples; }

  //! Get the replication multiples for modification
  std::vector<size_t>& Multiples() {
    this->validOutputDimensions = false;
    return multiples;
  }

  void ComputeOutputDimensions() override
  {
    if (multiples.size() > this->inputDimensions.size())
    {
      std::ostringstream oss;
      oss << "Replicate::ComputeOutputDimensions(): multiples vector must "
          << "have the same or less dimensions than InputDimensions";
      throw std::invalid_argument(oss.str());
    }

    size_t inputSize = this->inputDimensions[0];
    for (size_t i = 1; i < this->inputDimensions.size(); i++)
    {
      inputSize *= this->inputDimensions[i];
    }
    arma::umat idxs = arma::regspace<arma::uvec>(0, inputSize-1);

    // Here, we are going to pre-compute the source index for each output
    // for a single tensor.  Since the tensors are flattened into 1-d
    // vectors, we can fill the output row-wise based on these
    // indices.
    this->outputDimensions = this->inputDimensions;
    size_t sizeMult = 1;
    size_t outSize = 1;

    // iteratively reshape the index matrix such that the dimension
    // to be replicated (and all prior) are flattened to a column, and
    // then repmat columnwise.
    for (size_t i = 0; i < multiples.size(); i++)
    {
      if (multiples[i] != 1)
      {
        idxs.reshape(outSize * this->inputDimensions[i],
                     idxs.n_elem / (outSize * this->inputDimensions[i]));
        idxs = arma::repmat(idxs, multiples[i], 1);
        this->outputDimensions[i] *= multiples[i];
        sizeMult *= multiples[i];
      }
      outSize *= this->outputDimensions[i];
    }
    outIdxs = idxs.as_col();

    // Now, we are going to pre-compute the contribution of each output
    // element to the input elements.  This will be used in the backward
    // pass with a simple matrix multiplication.
    coefs = arma::zeros<MatType>(inputSize, outSize);
    for (size_t i = 0; i < outIdxs.n_elem; i++)
    {
      coefs.at(outIdxs.at(i), i) = 1.0 / (typename MatType::elem_type) sizeMult;
    }
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Parameter to indicate number of times to replicate along each dimension
  std::vector<size_t> multiples;

  // Cache the target indices for a single tensor for use
  // in the forward pass.
  arma::uvec outIdxs;

  // Cache the contributions of each output element to the
  // input elements for use in the backward pass.
  MatType coefs;
}; // class ReplicateType.

// Standard Replicate layer.
typedef ReplicateType<arma::mat> Replicate;

} // namespace mlpack

// Include implementation.
#include "replicate_impl.hpp"

#endif
