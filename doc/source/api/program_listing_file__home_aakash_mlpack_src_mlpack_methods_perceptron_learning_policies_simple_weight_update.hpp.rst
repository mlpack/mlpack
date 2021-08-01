
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_perceptron_learning_policies_simple_weight_update.hpp:

Program Listing for File simple_weight_update.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_perceptron_learning_policies_simple_weight_update.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/perceptron/learning_policies/simple_weight_update.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef _MLPACK_METHODS_PERCEPTRON_LEARNING_POLICIES_SIMPLE_WEIGHT_UPDATE_HPP
   #define _MLPACK_METHODS_PERCEPTRON_LEARNING_POLICIES_SIMPLE_WEIGHT_UPDATE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace perceptron {
   
   class SimpleWeightUpdate
   {
    public:
     template<typename VecType>
     void UpdateWeights(const VecType& trainingPoint,
                        arma::mat& weights,
                        arma::vec& biases,
                        const size_t incorrectClass,
                        const size_t correctClass,
                        const double instanceWeight = 1.0)
     {
       weights.col(incorrectClass) -= instanceWeight * trainingPoint;
       biases(incorrectClass) -= instanceWeight;
   
       weights.col(correctClass) += instanceWeight * trainingPoint;
       biases(correctClass) += instanceWeight;
     }
   };
   
   } // namespace perceptron
   } // namespace mlpack
   
   #endif
