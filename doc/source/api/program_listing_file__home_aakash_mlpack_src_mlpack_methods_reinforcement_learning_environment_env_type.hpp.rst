
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_env_type.hpp:

Program Listing for File env_type.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_env_type.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/environment/env_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ENVIRONMENT_ENV_TYPE_HPP
   #define MLPACK_METHODS_RL_ENVIRONMENT_ENV_TYPE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace rl {
   
   class DiscreteActionEnv
   {
    public:
     class State
     {
      public:
       State() : data(dimension)
       { /* Nothing to do here. */ }
   
       State(const arma::colvec& data) : data(data)
       { /* Nothing to do here */ }
   
       arma::colvec& Data() { return data; }
   
       const arma::colvec& Encode() const { return data; }
   
       static size_t dimension;
   
      private:
       arma::colvec data;
     };
   
     class Action
     {
      public:
       // To store the action.
       size_t action = 0;
       // Track the size of the action space.
       static size_t size;
     };
   
     double Sample(const State& /* state */,
                   const Action& /* action */,
                   State& /* nextState*/)
     { return 0; }
   
     State InitialSample() { return State(); }
     bool IsTerminal(const State& /* state */) const { return false; }
   };
   
   class ContinuousActionEnv
   {
    public:
     class State
     {
      public:
       State() : data(dimension)
       { /* Nothing to do here. */ }
   
       State(const arma::colvec& data) : data(data)
       { /* Nothing to do here */ }
   
       arma::colvec& Data() { return data; }
   
       const arma::colvec& Encode() const { return data; }
   
       static size_t dimension;
   
      private:
       arma::colvec data;
     };
   
     class Action
     {
      public:
       std::vector<double> action;
       // Storing degree of freedom.
       static size_t size;
   
       Action() : action(ContinuousActionEnv::Action::size)
       { /* Nothing to do here */ }
     };
   
     double Sample(const State& /* state */,
                   const Action& /* action */,
                   State& /* nextState*/)
     { return 0; }
   
     State InitialSample() { return State(); }
     bool IsTerminal(const State& /* state */) const { return false; }
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
