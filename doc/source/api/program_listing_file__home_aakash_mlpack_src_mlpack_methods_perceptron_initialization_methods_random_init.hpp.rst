
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_perceptron_initialization_methods_random_init.hpp:

Program Listing for File random_init.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_perceptron_initialization_methods_random_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/perceptron/initialization_methods/random_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PERCEPTRON_INITIALIZATION_METHODS_RANDOM_INIT_HPP
   #define MLPACK_METHODS_PERCEPTRON_INITIALIZATION_METHODS_RANDOM_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace perceptron {
   
   class RandomInitialization
   {
    public:
     RandomInitialization() { }
   
     inline static void Initialize(arma::mat& weights,
                                   arma::vec& biases,
                                   const size_t numFeatures,
                                   const size_t numClasses)
     {
       weights.randu(numFeatures, numClasses);
       biases.randu(numClasses);
     }
   }; // class RandomInitialization
   
   } // namespace perceptron
   } // namespace mlpack
   
   #endif
