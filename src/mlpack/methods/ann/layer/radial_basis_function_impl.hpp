/**
 * @file radial_basis_function_impl.hpp
 * @author Himanshu Pathak
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RADIAL_BASIS_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RADIAL_BASIS_FUNCTION_IMPL_HPP

// In case it hasn't yet been included.
#include "radial_basis_function.hpp"

namespace mlpack {

template<typename MatType, typename Activation>
RBFType<MatType, Activation>::RBFType() :
    Layer<MatType>(),
    outSize(0),
    betas(0)
{
  // Nothing to do here.
}

template<typename MatType, typename Activation>
RBFType<MatType, Activation>::RBFType(
    const size_t outSize,
    MatType& centres,
    double betas) :
    Layer<MatType>(),
    outSize(outSize),
    betas(betas),
    centres(centres)
{
  double sigmas = 0;
  if (betas == 0)
  {
    for (size_t i = 0; i < centres.n_cols; i++)
    {
      double maxDis = 0;
      MatType temp = centres.each_col() - centres.col(i);
      maxDis = max(pow(sum(pow((temp), 2), 0), 0.5).t());
      if (maxDis > sigmas)
        sigmas = maxDis;
    }
    this->betas = std::pow(2 * outSize, 0.5) / sigmas;
  }
}

template<typename MatType,
         typename Activation>
RBFType<MatType, Activation>::RBFType(const RBFType& other) :
    Layer<MatType>(other),
    outSize(other.outSize),
    betas(other.betas),
    centres(other.centres)
{
  // Nothing to do.
}

template<typename MatType,
         typename Activation>
RBFType<MatType, Activation>::RBFType(RBFType&& other) :
    Layer<MatType>(other),
    outSize(other.outSize),
    betas(other.betas),
    centres(std::move(other.centres))
{
  // Nothing to do.
}

template<typename MatType, typename Activation>
RBFType<MatType, Activation>&
RBFType<MatType, Activation>::operator=(const RBFType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    outSize = other.outSize;
    betas = other.betas;
    centres = other.centres;
  }

  return *this;
}

template<typename MatType, typename Activation>
RBFType<MatType, Activation>&
RBFType<MatType, Activation>::operator=(RBFType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    outSize = std::move(other.outSize);
    betas = std::move(other.betas);
    centres = std::move(other.centres);
  }

  return *this;
}

template<typename MatType, typename Activation>
void RBFType<MatType, Activation>::Forward(
    const MatType& input,
    MatType& output)
{
  // Sanity check: make sure the dimensions are right.
  if (input.n_rows != centres.n_rows)
  {
    Log::Fatal << "RBFType::Forward(): input size (" << input.n_rows << ") does"
        << " not match given center size (" << centres.n_rows << ")!"
        << std::endl;
  }

  distances = MatType(outSize, input.n_cols);

  for (size_t i = 0; i < input.n_cols; i++)
  {
    MatType temp = centres.each_col() - input.col(i);
    distances.col(i) = pow(sum(pow((temp), 2), 0), 0.5).t();
  }
  Activation::Fn(distances * std::pow(betas, 0.5), output);
}


template<typename MatType, typename Activation>
void RBFType<MatType, Activation>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& /* gy */,
    MatType& /* g */)
{
  // Nothing to do here.
}

template<typename MatType, typename Activation>
void RBFType<MatType, Activation>::ComputeOutputDimensions()
{
  this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(), 1);

  // This flattens the input.
  this->outputDimensions[0] = outSize;
}

template<typename MatType, typename Activation>
template<typename Archive>
void RBFType<MatType, Activation>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(distances));
  ar(CEREAL_NVP(centres));
  ar(CEREAL_NVP(betas));
}

} // namespace mlpack

#endif
