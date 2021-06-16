/**
 * @file radial_basis_function_impl.hpp
 * @author Himanshu Pathak
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RBF_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RBF_IMPL_HPP

// In case it hasn't yet been included.
#include "radial_basis_function.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType,
         typename Activation>
RBF<InputType, OutputType, Activation>::RBF() :
    outSize(0),
    betas(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType,
         typename Activation>
RBF<InputType, OutputType, Activation>::RBF(
    const size_t outSize,
    InputType& centres,
    double betas) :
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
      InputType temp = centres.each_col() - centres.col(i);
      maxDis = arma::accu(arma::max(arma::pow(arma::sum(
          arma::pow((temp), 2), 0), 0.5).t()));
      if (maxDis > sigmas)
        sigmas = maxDis;
    }
    this->betas = std::pow(2 * outSize, 0.5) / sigmas;
  }
}

template<typename InputType, typename OutputType, typename Activation>
void RBF<InputType, OutputType, Activation>::Forward(
    const InputType& input,
    OutputType& output)
{
  // Sanity check: make sure the dimensions are right.
  if (input.n_rows != centres.n_rows)
  {
    Log::Fatal << "RBF::Forward(): input size (" << input.n_rows << ") does "
        << "not match given center size (" << centres.n_rows << ")!"
        << std::endl;
  }

  distances = InputType(outSize, input.n_cols);

  for (size_t i = 0; i < input.n_cols; i++)
  {
    InputType temp = centres.each_col() - input.col(i);
    distances.col(i) = arma::pow(arma::sum(
        arma::pow((temp), 2), 0), 0.5).t();
  }
  Activation::Fn(distances * std::pow(betas, 0.5), output);
}


template<typename InputType, typename OutputType, typename Activation>
void RBF<InputType, OutputType, Activation>::Backward(
    const InputType& /* input */,
    const OutputType& /* gy */,
    OutputType& /* g */)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType, typename Activation>
template<typename Archive>
void RBF<InputType, OutputType, Activation>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(distances));
  ar(CEREAL_NVP(centres));
  ar(CEREAL_NVP(betas);
}

} // namespace ann
} // namespace mlpack

#endif
