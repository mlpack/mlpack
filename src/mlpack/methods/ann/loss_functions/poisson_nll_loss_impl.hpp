/**
 * @file poisson_nll_loss_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Poisson NegativeLogLikelihood class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_POISSON_NLL_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_POISSON_NLL_LOSS_IMPL_HPP

#define _USE_MATH_DEFINES
#include <math.h>

// In case it hasn't yet been included.
#include "poisson_nll_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
PoissonNLLLoss<InputDataType, OutputDataType>::PoissonNLLLoss(const bool log_input, const bool full, 
                                                              const double eps, const int reduction):
    log_input(log_input), full(full), eps(eps), reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double PoissonNLLLoss<InputDataType, OutputDataType>::Forward(const InputType&& input, TargetType&& target)
{
    auto loss = arma::zeros<InputType>(input.n_rows, input.n_cols);
    if(log_input)
        loss = arma::exp(input) - target%input; //element-wise multiplication of 
                                                //target and input
    else
        loss = input - target%arma::log(input + eps);
    
    if(full)
    {
        //TODO: needs to be vectorized.
        for(size_t i = 0; i < input.n_cols; ++i)
        {
            for(size_t j = 0; j < input.n_rows; ++i)
            {
                if(target(i,j)>1)
                    loss(i,j) += target(i,j)*std::log(target(i,j)) - target(i,j) + std::log(2*M_PI*target(i,j));
            }
        }   
    }
    if(reduction == 0)
        return arma::sum(loss);
    else if(reduction == 1)
        return arma::mean(loss);
    else
        throw std::invalid_argument("Reduction: 0 for sum and 1 for mean");
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void PoissonNLLLoss<InputDataType, OutputDataType>::Backward(
      const InputType&& input,
      const TargetType&& target,
      OutputType&& output)
{
   output = 1 - target/input;
   if(full)
   {
       output += arma::log(target) + 1/(2*target);
   }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void PoissonNLLLoss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
