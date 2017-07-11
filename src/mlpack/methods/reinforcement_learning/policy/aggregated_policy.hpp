#ifndef MLPACK_METHODS_RL_POLICY_AGGREGATED_POLICY_HPP
#define MLPACK_METHODS_RL_POLICY_AGGREGATED_POLICY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

template <typename PolicyType>
class AggregatedPolicy
{
public:
  using ActionType = typename PolicyType::ActionType;
  AggregatedPolicy(std::vector<PolicyType> policies,
                   arma::colvec distribution) :
          policies(std::move(policies)),
          distribution(std::move(distribution))
  { };

  ActionType Sample(const arma::colvec& actionValue, bool deterministic = false)
  {
    if (deterministic)
      return policies.front().Sample(actionValue, true);
    return policies[RandomIndex(distribution)].Sample(actionValue, false);
  }

  void Anneal()
  {
    for (auto& policy : policies)
      policy.Anneal();
  }

private:
  std::vector<PolicyType> policies;
  arma::colvec distribution;

};

}
}

#endif