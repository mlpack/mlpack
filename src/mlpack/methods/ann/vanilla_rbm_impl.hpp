/**
 * @file vanilla_rbm_impl.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_VANILLA_RBM_IMPL_HPP
#define MLPACK_METHODS_ANN_VANILLA_RBM_IMPL_HPP

// In case it hasn't been included yet.
#include "vanilla_rbm.hpp"

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

#include "layer/layer.hpp"
#include "layer/base_layer.hpp"
#include "visitor/weight_set_visitor.hpp"
#include "visitor/delete_visitor.hpp"
#include "visitor/forward_visitor.hpp"
#include "visitor/weight_size_visitor.hpp"
#include "init_rules/gaussian_init.hpp"
#include "visitor/reset_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


// Intialise a linear layer and sigmoid layer
template<typename InitializationRuleType, typename VisibleLayerType, typename HiddenLayerType>
VanillaRBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::VanillaRBM(
  InitializationRuleType initializeRule, 
  VisibleLayerType visible, 
  HiddenLayerType hidden):
  visible(visible), hidden(hidden), initializeRule(initializeRule)
{/*Nothing to do here */}

template<typename InitializationRuleType, typename VisibleLayerType, typename HiddenLayerType>
void VanillaRBM<InitializationRuleType, VisibleLayerType, HiddenLayerType>::Reset()
{
  // Since we are dealing with a static stucture no need for visitor
  // Reset the parameters of all the layers
  size_t weight = 0;
  weight+= visible.Parameters().n_elem;

  parameter.set_size(weight, 1);
  initializeRule.Initialize(parameter, parameter.n_elem, 1);
  visible.Parameters() = arma::mat(parameter.memptr(), 1, weight, false, false);
  // set the hidden parameters to point to visible layers switch the biases parameters
  hidden.Parameters() = arma::mat(visible.Parameters().memptr(), 1, visible.Parameters().n_elem, false, false);

  
  visible.Reset();
  hidden.Reset();

}
/*
template<typename InitializationRuleType,
         typename VisibleLayerType,
         typename HiddenLayerType>
template< template<typename, typename...> class Optimizer, typename... OptimizerTypeArgs>
void VanillaRBM<InitializationRuleType, 
           VisibleLayerType, 
           HiddenLayerType>::Train(const arma::mat& predictors, 
                                   Optimizer<NetworkType, 
                                   OptimizerTypeArgs...>& optimizer)
{

  numFunctions = predictors.n_cols;

  this->predictors = std::move(predictors);

  if (!reset)
  {
    ResetParameters();
  }

  // Train the model.
  Timer::Start("rbm_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("rbm_optimization");
}
*/
/* 
* This function calculates
* the free energy of the model
*/
template<typename InitializationRuleType,
         typename VisibleLayerType,
         typename HiddenLayerType>
double VanillaRBM<InitializationRuleType, 
           VisibleLayerType, 
           HiddenLayerType>::FreeEnergy(const arma::mat&& input)
{
  visible.FreeEnergy(input);
}

template<typename InitializationRuleType, typename VisibleLayerType, typename HiddenLayerType>
void VanillaRBM<InitializationRuleType, VisibleLayerType, 
  HiddenLayerType>::SampleHidden(arma::mat&& input, arma::mat&& output)
{
  visible.Sample(std::move(input), std::move(output));
}

template<typename InitializationRuleType, typename VisibleLayerType, typename HiddenLayerType>
void VanillaRBM<InitializationRuleType, VisibleLayerType, 
  HiddenLayerType>::SampleVisible(arma::mat&& input, arma::mat&& output)
{
  hidden.Sample(std::move(input), std::move(output));
}

template<typename InitializationRuleType, typename VisibleLayerType, typename HiddenLayerType>
void VanillaRBM<InitializationRuleType, VisibleLayerType, 
  HiddenLayerType>::Gibbs(arma::mat&& input, arma::mat&& negative_sample, size_t num_steps , bool persistence)
{
  arma::mat temp1;

  if(persistence && !state.is_empty())
    input = state;
  
  for(size_t j = 0; j < num_steps; j++)
  {
    SampleHidden(std::move(input), std::move(temp1));
    SampleVisible(std::move(temp1), std::move(input));
  }

  state = input;
  negative_sample = state;

}

template<typename InitializationRuleType, typename VisibleLayerType, typename HiddenLayerType>
void VanillaRBM<InitializationRuleType, VisibleLayerType, 
  HiddenLayerType>::Gradient(arma::mat input, const size_t k, const bool persistence, arma::mat& output)
{
    arma::mat positive_gradient, negative_gradient, negative_sample;
    // Collect the positive gradients
    CalcGradient(std::move(input), positive_gradient);

    // Collect the negative samples
    Gibbs(input, negative_sample, k, persistence);

    // Collect the negative gradients
    CalcGradient(std::move(negative_sample), negative_gradient);

    output = positive_gradient - negative_gradient;
}

//! Serialize the model.
template<typename InitializationRuleType,
         typename VisibleLayerType,
         typename HiddenLayerType>
template<typename Archive>
void VanillaRBM<InitializationRuleType, 
           VisibleLayerType, 
           HiddenLayerType>::Serialize(Archive& ar, const unsigned int /* version */)
{
  // Todo: Save other parameters 
  ar & data::CreateNVP(parameter, "parameter");

}

} /** Artificial Neural Network. */ 
} // namespace mlpack
#endif