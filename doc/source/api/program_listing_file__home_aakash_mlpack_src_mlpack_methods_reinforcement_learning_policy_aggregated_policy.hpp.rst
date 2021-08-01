
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_policy_aggregated_policy.hpp:

Program Listing for File aggregated_policy.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_policy_aggregated_policy.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/policy/aggregated_policy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_POLICY_AGGREGATED_POLICY_HPP
   #define MLPACK_METHODS_RL_POLICY_AGGREGATED_POLICY_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/dists/discrete_distribution.hpp>
   
   namespace mlpack {
   namespace rl {
   
   template <typename PolicyType>
   class AggregatedPolicy
   {
    public:
     using ActionType = typename PolicyType::ActionType;
   
     AggregatedPolicy(std::vector<PolicyType> policies,
                      const arma::colvec& distribution) :
         policies(std::move(policies)),
         sampler({distribution})
     { /* Nothing to do here. */ };
   
     ActionType Sample(const arma::colvec& actionValue, bool deterministic = false)
     {
       if (deterministic)
         return policies.front().Sample(actionValue, true);
       size_t selected = arma::as_scalar(sampler.Random());
       return policies[selected].Sample(actionValue, false);
     }
   
     void Anneal()
     {
       for (PolicyType& policy : policies)
         policy.Anneal();
     }
   
    private:
     std::vector<PolicyType> policies;
   
     distribution::DiscreteDistribution sampler;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
