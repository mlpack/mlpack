/**
 * @file contrastive_loss_impl.hpp
 * @author Shardul Shailendra Parab
 *
 * Implementation of the contrastive loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CONTRASTIVE_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CONTRASTIVE_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "contrastive_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
ContrastiveLoss<InputDataType, OutputDataType>::ContrastiveLoss(
  const double margin, const double lambda) : margin(margin), lambda(lambda)
  
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double ContrastiveLoss<InputDataType, OutputDataType>::Forward(
  const InputType&& input1, const InputType&& input2, const TargetType&& target){

  //calculating the sum of the distances along the row dimension and then calculating its square root
  arma::mat distances = sqrt(sum(square(input1 - input2), 0));
  
  //in reference to the link https://leimao.github.io/article/Siamese-Network-MNIST/ due to poor gradient properties
  //we add a value lamda = 10 ^ -6
  arma::mat distances_prime = sqrt(square(distances) + lambda);

  arma::mat margin_vector = (margin * arma::ones(1, distances.n_cols)) - distances_prime;

  //checking the condition whether margin - distance is greater than or less than zero
  //if less than zero, replace with zero
  margin_vector.for_each([] (arma::mat::elem_type& val) {if(val<0) val = 0;});

  //returning the total error
  return arma::accu( (0.5 * ((1 - target) % square(distances))) + (0.5 * (target % square(margin_vector))));
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void ContrastiveLoss<InputDataType, OutputDataType>::Backward(
    const InputType&& input1,
    const InputType&& input2,
    const TargetType&& target,
    OutputType&& output)
    {
      
      arma::mat distances_prime = sqrt(sum(square(input1 - input2),0) + lambda);

      arma::mat result;
      result.set_size(1,distances_prime.n_cols);

      arma::mat margin_vector = (margin * arma::ones(1,distances_prime.n_cols));

      for(size_t i = 0; i<margin_vector.n_cols; i++){
        if(margin_vector(0,i) < distances_prime(0,i))
            result(0,i) = target(0,i) * distances_prime(0,i);
        else
            result(0,i) = (target(0,i) * distances_prime(0,i)) - ((1 - target(0,i)) * (margin_vector(0,i) - distances_prime(0,i))); 
    }

    output = result;

    }

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ContrastiveLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{

}

}//namespace ann
}//namespace mlpack

#endif

