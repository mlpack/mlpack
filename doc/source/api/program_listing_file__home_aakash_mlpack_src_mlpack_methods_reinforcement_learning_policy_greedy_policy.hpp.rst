
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_policy_greedy_policy.hpp:

Program Listing for File greedy_policy.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_policy_greedy_policy.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_POLICY_GREEDY_POLICY_HPP
   #define MLPACK_METHODS_RL_POLICY_GREEDY_POLICY_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace rl {
   
   template <typename EnvironmentType>
   class GreedyPolicy
   {
    public:
     using ActionType = typename EnvironmentType::Action;
   
     GreedyPolicy(const double initialEpsilon,
                  const size_t annealInterval,
                  const double minEpsilon,
                  const double decayRate = 1.0) :
         epsilon(initialEpsilon),
         minEpsilon(minEpsilon),
         delta(((initialEpsilon - minEpsilon) * decayRate) / annealInterval)
     { /* Nothing to do here. */ }
   
     ActionType Sample(const arma::colvec& actionValue,
                       bool deterministic = false,
                       const bool isNoisy = false)
     {
       double exploration = math::Random();
       ActionType action;
   
       // Select the action randomly.
       if (!deterministic && exploration < epsilon && isNoisy == false)
       {
         action.action = static_cast<decltype(action.action)>
             (math::RandInt(ActionType::size));
       }
       // Select the action greedily.
       else
       {
         action.action = static_cast<decltype(action.action)>(
             arma::as_scalar(arma::find(actionValue == actionValue.max(), 1)));
       }
       return action;
     }
   
     void Anneal()
     {
       epsilon -= delta;
       epsilon = std::max(minEpsilon, epsilon);
     }
   
     const double& Epsilon() const { return epsilon; }
   
    private:
     double epsilon;
   
     double minEpsilon;
   
     double delta;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
