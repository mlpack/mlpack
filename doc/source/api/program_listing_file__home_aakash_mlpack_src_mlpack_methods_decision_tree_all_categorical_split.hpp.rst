
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_decision_tree_all_categorical_split.hpp:

Program Listing for File all_categorical_split.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_decision_tree_all_categorical_split.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/decision_tree/all_categorical_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DECISION_TREE_ALL_CATEGORICAL_SPLIT_HPP
   #define MLPACK_METHODS_DECISION_TREE_ALL_CATEGORICAL_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree {
   
   template<typename FitnessFunction>
   class AllCategoricalSplit
   {
    public:
     // No extra info needed for split.
     class AuxiliarySplitInfo { };
   
     template<bool UseWeights, typename VecType, typename LabelsType,
              typename WeightVecType>
     static double SplitIfBetter(
         const double bestGain,
         const VecType& data,
         const size_t numCategories,
         const LabelsType& labels,
         const size_t numClasses,
         const WeightVecType& weights,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         arma::vec& splitInfo,
         AuxiliarySplitInfo& aux);
   
     template<bool UseWeights, typename VecType, typename ResponsesType,
              typename WeightVecType>
     static double SplitIfBetter(
         const double bestGain,
         const VecType& data,
         const size_t numCategories,
         const ResponsesType& responses,
         const WeightVecType& weights,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         double& splitInfo,
         AuxiliarySplitInfo& aux,
         FitnessFunction& fitnessFunction);
   
     static size_t NumChildren(const double& splitInfo,
                               const AuxiliarySplitInfo& /* aux */);
   
     template<typename ElemType>
     static size_t CalculateDirection(
         const ElemType& point,
         const double& splitInfo,
         const AuxiliarySplitInfo& /* aux */);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "all_categorical_split_impl.hpp"
   
   #endif
