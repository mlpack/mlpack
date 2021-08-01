
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_env_type.cpp:

Program Listing for File env_type.cpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_env_type.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/environment/env_type.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "env_type.hpp"
   
   namespace mlpack {
   namespace rl {
   
   // Instantiate static members.
   
   size_t DiscreteActionEnv::State::dimension = 0;
   size_t DiscreteActionEnv::Action::size = 0;
   
   size_t ContinuousActionEnv::State::dimension = 0;
   size_t ContinuousActionEnv::Action::size = 0;
   
   } // namespace rl
   } // namespace mlpack
