
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_decision_tree_best_binary_numeric_split.hpp:

Program Listing for File best_binary_numeric_split.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_decision_tree_best_binary_numeric_split.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/decision_tree/best_binary_numeric_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DECISION_TREE_BEST_BINARY_NUMERIC_SPLIT_HPP
   #define MLPACK_METHODS_DECISION_TREE_BEST_BINARY_NUMERIC_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "mse_gain.hpp"
   
   #include <mlpack/core/util/sfinae_utility.hpp>
   
   namespace mlpack {
   namespace tree {
   
   // This gives us a HasBinaryGains<T, U> type (where U is a function pointer)
   // we can use with SFINAE to catch when a type has a BinaryGains(...) function.
   HAS_MEM_FUNC(BinaryGains, HasBinaryGains);
   
   // This struct will have `value` set to `true` if a BinaryGains() function of
   // the right signature is detected.  We only check for BinaryGains(), and not
   // BinaryScanInitialize() or BinaryStep(), because those two are template
   // members functions and would make this check far more difficult.
   //
   // The unused UseWeights template parameter is necessary to ensure that the
   // compiler thinks the result `value` depends on a parameter specific to the
   // SplitIfBetter() function in BestBinaryNumericSplit().
   template<typename T, bool /* UseWeights */>
   struct HasOptimizedBinarySplitForms
   {
     const static bool value = HasBinaryGains<T,
         std::tuple<double, double>(T::*)()>::value;
   };
   
   template<typename FitnessFunction>
   class BestBinaryNumericSplit
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
         AuxiliarySplitInfo& aux);
   
     template<bool UseWeights, typename VecType, typename ResponsesType,
              typename WeightVecType>
     static typename std::enable_if<
         !HasOptimizedBinarySplitForms<FitnessFunction, UseWeights>::value,
         double>::type
     SplitIfBetter(
         const double bestGain,
         const VecType& data,
         const ResponsesType& responses,
         const WeightVecType& weights,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         double& splitInfo,
         AuxiliarySplitInfo& aux,
         FitnessFunction& fitnessFunction);
   
     template<bool UseWeights, typename VecType, typename ResponsesType,
             typename WeightVecType>
     static typename std::enable_if<
         HasOptimizedBinarySplitForms<FitnessFunction, UseWeights>::value,
         double>::type
     SplitIfBetter(
         const double bestGain,
         const VecType& data,
         const ResponsesType& responses,
         const WeightVecType& weights,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         double& splitInfo,
         AuxiliarySplitInfo& /* aux */,
         FitnessFunction& fitnessFunction);
   
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
   #include "best_binary_numeric_split_impl.hpp"
   
   #endif
