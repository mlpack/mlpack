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
 * input a specified number of times along a each dimension.  The output
 * will have the same number of dimensions as the input, with each dimension
 * multiplied by a specified multiple.
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
   * Create the Repeat object, specifying the number of times to repeat
   * along each dimension.
   *
   * @param multiples The number of times to repeat along each axis. Must be
   *        the same size or smaller than InputDimensions.
   */
  RepeatType(std::vector<size_t> multiples);

  /**
   * Destroy the layers held by the model.
   */
  virtual ~RepeatType() { }

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
   * @param * (input) The input data (x) given to the forward pass.
   * @param * (output) The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g) override;

  //! Get the repeat multiples
  const std::vector<size_t>& Multiples() const { return multiples; }

  //! Get the repeat multiples for modification
  std::vector<size_t>& Multiples()
  {
    this->validOutputDimensions = false;
    return multiples;
  }

  /**
   * @brief Computes the output dimensions of the Repeat layer.
   *
   * The ComputeOutputDimensions function computes the output dimensions of the
   * Repeat layer based on the input dimensions and the repeat multiples. The
   * output dimensions will have the same number of dimensions as the input,
   * with all dimensions other than the one specified in the axis being the same
   * size as the input.
   */
  void ComputeOutputDimensions() override
  {
    if (multiples.size() > this->inputDimensions.size())
    {
      std::ostringstream oss;
      oss << "Repeat::ComputeOutputDimensions(): multiples vector must "
          << "have the same or fewer dimensions than InputDimensions";
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
    // to be repeated (and all prior) are flattened to a column, and
    // then repelem rowwise.
    for (size_t i = 0; i < multiples.size(); i++)
    {
      if (multiples[i] != 1)
      {
        // For the first dimension, we need to do the repelem columnwise.
        if (i == 0)
        {
          idxs.reshape(outSize * this->inputDimensions[i],
                       idxs.n_elem / (outSize * this->inputDimensions[i]));
          idxs = arma::repelem(idxs, multiples[i], 1);
        }
        else
        {
          idxs.reshape(outSize,
                       idxs.n_elem / outSize);
          idxs = arma::repelem(idxs, 1, multiples[i]);
        }
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
  //! Parameter to indicate number of times to repeat along each dimension
  std::vector<size_t> multiples;

  // Cache the target indices for a single tensor for use
  // in the forward pass.
  arma::uvec outIdxs;

  // Cache the contributions of each output element to the
  // input elements for use in the backward pass.
  MatType coefs;
}; // class RepeatType.

// Standard Repeat layer.
typedef RepeatType<arma::mat> Repeat;

} // namespace mlpack

// Include implementation.
#include "repeat_impl.hpp"

#endif
