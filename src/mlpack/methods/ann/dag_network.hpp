#ifndef MLPACK_METHODS_ANN_DAG_NETWORK_HPP
#define MLPACK_METHODS_ANN_DAG_NETWORK_HPP

#include "init_rules/init_rules.hpp"

namespace mlpack {

template<
    typename InitializationRuleType = RandomInitialization,
    typename MatType = arma::mat>
class DAGNetwork 
{
public:
  DAGNetwork(InitializationRuleType initializeRule = InitializationRuleType());

  DAGNetwork(const DAGNetwork& other);
  DAGNetwork(DAGNetwork&& other);
  DAGNetwork& operator=(const DAGNetwork& other);
  DAGNetwork& operator=(DAGNetwork&& other);

private:
  InitializationRuleType initializeRule;
};

} // namespace mlpack

#include "dag_network_impl.hpp"

#endif
