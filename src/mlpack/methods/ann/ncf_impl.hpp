/**
 * @file ncf_impl.hpp
 * @author Haritha Nair
 *
 * Definition of the NCFNetwork class, which implements feed forward networks
 * for neural collaborative filtering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_NCF_IMPL_HPP
#define MLPACK_METHODS_ANN_NCF_IMPL_HPP

// In case it hasn't been included yet.
#include "ncf.hpp"

#include "visitor/forward_visitor.hpp"
#include "visitor/backward_visitor.hpp"
#include "visitor/deterministic_set_visitor.hpp"
#include "visitor/gradient_set_visitor.hpp"
#include "visitor/gradient_visitor.hpp"
#include "visitor/set_input_height_visitor.hpp"
#include "visitor/set_input_width_visitor.hpp"

#include <boost/serialization/variant.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename Model, typename InitializationRuleType>
NCFNetwork<Model, InitializationRuleType>::NCFNetwork(
    Model& userModel, Model& itemModel,
    InitializationRuleType initializeRule) :
    userModel(userModel),
    itemModel(itemModel),
    initializeRule(std::move(initializeRule))
{
  /* Nothing to do here */
}

template<typename Model, typename InitializationRuleType>
void NCFNetwork<Model, InitializationRuleType>::ResetData(arma::mat user,
                                                          arma::mat item,
                                                          arma::mat responses)
{
  numFunctions = responses.n_cols;
  this->user = std::move(user);
  this->item = std::move(item);
  this->responses = std::move(responses);
  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
    ResetParameters();
}

template<typename Model, typename InitializationRuleType>
template<typename OptimizerType>
void NCFNetwork<Model, InitializationRuleType>::Train(arma::mat user,
                                                      arma::mat item,
                                                      arma::mat responses,
                                                      OptimizerType& optimizer)
{
  ResetData(std::move(user), std::move(item), std::move(responses));

  // Train the model.
  Timer::Start("ncf_optimization");
  const double out = optimizer.Optimize(*this, parameter);
  Timer::Stop("ncf_optimization");

  Log::Info << "NCFNetwork::NCFNetwork(): final objective of trained model is "
      << out << "." << std::endl;
}

template<typename Model, typename InitializationRuleType>
void NCFNetwork<Model, InitializationRuleType>::Forward(arma::mat userInput,
                                                        arma::mat itemInput,
                                                        arma::mat& results)
{
  if (parameter.is_empty())
    ResetParameters();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  curUserInput = std::move(userInput);
  curItemInput = std::move(itemInput);
  Forward(std::move(curUserInput), std::move(curUserInput));
  results = boost::apply_visitor(outputParameterVisitor, network.back());
}

template<typename Model, typename InitializationRuleType>
void NCFNetwork<Model, InitializationRuleType>::Predict(arma::mat user,
                                                        arma::mat item,
                                                        arma::mat& results)
{
  if (parameter.is_empty())
    ResetParameters();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  arma::mat resultsTemp;
  Forward(std::move(arma::mat(user.colptr(0),
      user.n_rows, 1, false, true)), std::move(arma::mat(item.colptr(0),
      item.n_rows, 1, false, true)));
  resultsTemp = boost::apply_visitor(outputParameterVisitor,
      network.back()).col(0);

  results = arma::mat(resultsTemp.n_elem, user.n_cols);
  results.col(0) = resultsTemp.col(0);

  for (size_t i = 1; i < user.n_cols; i++)
  {
    Forward(std::move(arma::mat(user.colptr(i),
        user.n_rows, 1, false, true)), std::move(arma::mat(item.colptr(i),
        item.n_rows, 1, false, true)));

    resultsTemp = boost::apply_visitor(outputParameterVisitor,
        network.back());
    results.col(i) = resultsTemp.col(0);
  }
}

template<typename Model, typename InitializationRuleType>
double NCFNetwork<Model, InitializationRuleType>::Evaluate(
    const arma::mat& parameters)
{
  double res = 0;
  for (size_t i = 0; i < users.n_cols; ++i)
    res += Evaluate(parameters, i, true);

  return res;
}

template<typename Model, typename InitializationRuleType>
double NCFNetwork<Model, InitializationRuleType>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t begin,
    const size_t batchSize,
    const bool deterministic)
{
  if (parameter.is_empty())
    ResetParameters();

  if (deterministic != this->deterministic)
  {
    this->deterministic = deterministic;
    ResetDeterministic();
  }

  Forward(std::move(user.cols(begin, begin + batchSize - 1)),
      std::move(item.cols(begin, begin + batchSize - 1)));
  double userRes = userModel.outputLayer.Forward(std::move(
      boost::apply_visitor(outputParameterVisitor, userModel.network.back())),
      std::move(responses.cols(begin, begin + batchSize - 1)));

  double itemRes = itemModel.outputLayer.Forward(std::move(
      boost::apply_visitor(outputParameterVisitor, itemModel.network.back())),
      std::move(responses.cols(begin, begin + batchSize - 1)));

  // Multiplymerge to be implemented.

  return res;
}

template<typename Model, typename InitializationRuleType>
void NCFNetwork<Model, InitializationRuleType>::Gradient(
    const arma::mat& parameters,
    const size_t begin,
    arma::mat& gradient,
    const size_t batchSize)
{
  // To be implemented
}

template<typename Model, typename InitializationRuleType>
void NCFNetwork<Model, InitializationRuleType>::Shuffle()
{
  math::ShuffleData(user, item, responses, user, item, responses);
}

template<typename Model, typename InitializationRuleType>
void NCFNetwork<Model, InitializationRuleType>::ResetParameters()
{
  ResetDeterministic();

  // Reset the network parameter with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(userModel.network, parameter);
  networkInit.Initialize(itemModel.network, parameter);
}

template<typename Model, typename InitializationRuleType>
void NCFNetwork<Model, InitializationRuleType>::ResetDeterministic()
{
  DeterministicSetVisitor deterministicSetVisitor(deterministic);
  std::for_each(userModel.network.begin(), userModel.network.end(),
      boost::apply_visitor(deterministicSetVisitor));
  std::for_each(itemModel.network.begin(), itemModel.network.end(),
      boost::apply_visitor(deterministicSetVisitor));
}

template<typename Model, typename InitializationRuleType>
void NCFNetwork<Model, InitializationRuleType>::ResetGradients(
    arma::mat& gradient)
{
  size_t offset = 0;
  for (size_t i = 0; i < network.size(); ++i)
  {
    offset += boost::apply_visitor(GradientSetVisitor(std::move(gradient),
        offset), network[i]);
  }
}

template<typename Model, typename InitializationRuleType>
void NCFNetwork<Model, InitializationRuleType>::Forward(
    arma::mat&& userInput,
    arma::mat&& itemInput)
{
  if (!reset)
    ResetData();

  userModel.Forward(std::move(userInput));
  itemModel.Forward(std::move(itemInput));
}

template<typename Model, typename InitializationRuleType>
template<typename Archive>
void NCFNetwork<Model, InitializationRuleType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(parameter);
  ar & BOOST_SERIALIZATION_NVP(width);
  ar & BOOST_SERIALIZATION_NVP(height);
  ar & BOOST_SERIALIZATION_NVP(curUserInput);
  ar & BOOST_SERIALIZATION_NVP(curItemInput);
  ar & BOOST_SERIALIZATION_NVP(userModel);
  ar & BOOST_SERIALIZATION_NVP(itemModel);
}

} // namespace ann
} // namespace mlpack

#endif
