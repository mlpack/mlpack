
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_perceptron_initialization_methods_zero_init.hpp:

Program Listing for File zero_init.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_perceptron_initialization_methods_zero_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/perceptron/initialization_methods/zero_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PERCEPTRON_INITIALIZATION_METHODS_ZERO_INIT_HPP
   #define MLPACK_METHODS_PERCEPTRON_INITIALIZATION_METHODS_ZERO_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace perceptron {
   
   class ZeroInitialization
   {
    public:
     ZeroInitialization() { }
   
     inline static void Initialize(arma::mat& weights,
                                   arma::vec& biases,
                                   const size_t numFeatures,
                                   const size_t numClasses)
     {
       weights.zeros(numFeatures, numClasses);
       biases.zeros(numClasses);
     }
   }; // class ZeroInitialization
   
   } // namespace perceptron
   } // namespace mlpack
   
   #endif
