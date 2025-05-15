#ifndef MLPACK_METHODS_ANN_DAG_NETWORK_IMPL_HPP
#define MLPACK_METHODS_ANN_DAG_NETWORK_IMPL_HPP

#include "dag_network.hpp"

namespace mlpack {

template<typename InitializationRuleType, 
         typename MatType>
DAGNetwork<
    InitializationRuleType, 
    MatType
>::DAGNetwork(InitializationRuleType initializeRule) :
    initializeRule(initializeRule)
{}

template<typename InitializationRuleType,
         typename MatType>
DAGNetwork<
    InitializationRuleType,
    MatType
>::DAGNetwork(const DAGNetwork& network) :
    initializeRule(network.initializeRule)
{}

template<typename InitializationRuleType,
         typename MatType>
DAGNetwork<
    InitializationRuleType,
    MatType
>::DAGNetwork(DAGNetwork&& network) :
    initializeRule(std::move(network.initializeRule))
{}

template<typename InitializationRuleType,
         typename MatType>
DAGNetwork<InitializationRuleType, MatType>& DAGNetwork<
    InitializationRuleType,
    MatType
>::operator=(const DAGNetwork& other)
{
  if (this != &other)
  {
    initializeRule = other.initializeRule;
  }

  return *this;
}

template<typename InitializationRuleType,
         typename MatType>
DAGNetwork<InitializationRuleType, MatType>& DAGNetwork<
    InitializationRuleType,
    MatType
>::operator=(DAGNetwork&& other)
{
  if (this != &other)
  {
    initializeRule = std::move(other.initializeRule);
  }

  return *this;
}

} // namespace mlpack

#endif
