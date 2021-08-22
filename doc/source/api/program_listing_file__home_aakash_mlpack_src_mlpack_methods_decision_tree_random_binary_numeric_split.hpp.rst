
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_decision_tree_random_binary_numeric_split.hpp:

Program Listing for File random_binary_numeric_split.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_decision_tree_random_binary_numeric_split.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/decision_tree/random_binary_numeric_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DECISION_TREE_RANDOM_BINARY_NUMERIC_SPLIT_HPP
   #define MLPACK_METHODS_DECISION_TREE_RANDOM_BINARY_NUMERIC_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree {
   
   template<typename FitnessFunction>
   class RandomBinaryNumericSplit
   {
    public:
     // No extra info needed for split.
     class AuxiliarySplitInfo { };
   
     template<bool UseWeights, typename VecType, typename WeightVecType>
     static double SplitIfBetter(
         const double bestGain,
         const VecType& data,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const WeightVecType& weights,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         arma::vec& splitInfo,
         AuxiliarySplitInfo& aux,
         const bool splitIfBetterGain = false);
   
     template<bool UseWeights, typename VecType, typename WeightVecType>
     static double SplitIfBetter(
         const double bestGain,
         const VecType& data,
         const arma::rowvec& responses,
         const WeightVecType& weights,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         double& splitInfo,
         AuxiliarySplitInfo& aux,
         FitnessFunction& fitnessFunction,
         const bool splitIfBetterGain = false);
   
     static size_t NumChildren(const double& /* splitInfo */,
                               const AuxiliarySplitInfo& /* aux */)
     {
       return 2;
     }
   
     template<typename ElemType>
     static size_t CalculateDirection(
         const ElemType& point,
         const double& splitInfo,
         const AuxiliarySplitInfo& /* aux */);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "random_binary_numeric_split_impl.hpp"
   
   #endif
