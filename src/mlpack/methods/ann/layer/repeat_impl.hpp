/**
 * @file methods/ann/layer/repeat_impl.hpp
 * @author Adam Kropp
 *
 * Implementation of the Repeat class, which repeats the input n times
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REPEAT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_REPEAT_IMPL_HPP

#include "repeat.hpp"

namespace mlpack {

template<typename MatType>
RepeatType<MatType>::RepeatType(
    std::vector<size_t> multiples, bool interleave) :
    Layer<MatType>(),
    multiples(std::move(multiples)),
    interleave(interleave)
{
  // Nothing to do.
}

template<typename MatType>
RepeatType<MatType>::RepeatType() :
    Layer<MatType>(),
    interleave(false)
{
  // Nothing to do.
}

template<typename MatType>
RepeatType<MatType>::RepeatType(const RepeatType& other) :
    Layer<MatType>(other),
    multiples(other.multiples),
    interleave(other.interleave),
    outIdxs(other.outIdxs),
    sizeMult(other.sizeMult)
{
  // Nothing else to do.
}

template<typename MatType>
RepeatType<MatType>::RepeatType(RepeatType&& other)  noexcept :
    Layer<MatType>(std::move(other)),
    multiples(other.multiples),
    interleave(other.interleave),
    outIdxs(other.outIdxs),
    sizeMult(other.sizeMult)
{
  // Nothing else to do.
}

template<typename MatType>
RepeatType<MatType>& RepeatType<MatType>::operator=(const RepeatType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    multiples = other.multiples;
    interleave = other.interleave;
    outIdxs = other.outIdxs;
    sizeMult = other.sizeMult;
  }

  return *this;
}

template<typename MatType>
RepeatType<MatType>& RepeatType<MatType>::operator=(RepeatType&& other) noexcept
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    multiples = std::move(other.multiples);
    interleave = std::move(other.interleave);
    outIdxs = std::move(other.outIdxs);
    sizeMult = other.sizeMult;
  }

  return *this;
}

template<typename MatType>
void RepeatType<MatType>::ComputeOutputDimensions()
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
  MatType idxs(inputSize, 1);
  for (size_t i=0; i<inputSize; i++)
  {
    idxs.at(i) = i;
  }
  // want to do this, but can't leave it without a namespace (e.g. arma::)
  //  MatType idxs = linspace<MatType>(0, inputSize - 1, inputSize);

  // Here, we are going to pre-compute the source index for each output
  // for a single tensor.  Since the tensors are flattened into 1-d
  // vectors, we can fill the output row-wise based on these
  // indices.
  this->outputDimensions = this->inputDimensions;
  sizeMult = 1;
  size_t outSize = 1;

  // iteratively reshape the index matrix such that the dimension
  // to be repeated (and all prior) are flattened to a column, and
  // then repelem rowwise or repmat columwise.
  for (size_t i = 0; i < multiples.size(); i++)
  {
    if (multiples[i] != 1)
    {
      if (interleave)
      {
        // For the first dimension, we need to do the repelem columnwise.
        if (i == 0)
        {
          idxs.reshape(outSize * this->inputDimensions[i],
                       idxs.n_elem / (outSize * this->inputDimensions[i]));
          idxs = repelem(idxs, multiples[i], 1);
        }
        else
        {
          idxs.reshape(outSize,
                       idxs.n_elem / outSize);
          idxs = repelem(idxs, 1, multiples[i]);
        }
      }
      else
      {
        idxs.reshape(outSize * this->inputDimensions[i],
                     idxs.n_elem / (outSize * this->inputDimensions[i]));
        idxs = repmat(idxs, multiples[i], 1);
      }
      this->outputDimensions[i] *= multiples[i];
      sizeMult *= multiples[i];
    }
    outSize *= this->outputDimensions[i];
  }
  outIdxs.resize(idxs.n_elem);
  for (size_t i=0; i<idxs.n_elem; i++)
  {
    outIdxs[i] = idxs.at(i);
  }
}

template<typename MatType>
void RepeatType<MatType>::Forward(const MatType& input, MatType& output)
{
#pragma omp parallel for
  for (size_t j=0; j<input.n_cols; j++)
  {
    for (size_t i = 0; i < outIdxs.size(); i++)
    {
      output.at(i, j) = input.at(outIdxs.at(i), j);
    }
  }
}

template<typename MatType>
void RepeatType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
#pragma omp parallel for
  for (size_t j=0; j<gy.n_cols; j++)
  {
    g.col(j).zeros();
    for (size_t i=0; i<outIdxs.size(); i++) {
      g.at(outIdxs.at(i), j) += gy.at(i, j);
    }
    g.col(j) /= sizeMult;
  }
}

template<typename MatType>
template<typename Archive>
void RepeatType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(multiples));
  ar(CEREAL_NVP(interleave));
  ar(CEREAL_NVP(outIdxs));
  ar(CEREAL_NVP(sizeMult));
}

} // namespace mlpack

#endif
