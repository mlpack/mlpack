
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_decision_tree_random_binary_numeric_split_impl.hpp:

Program Listing for File random_binary_numeric_split_impl.hpp
=============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_decision_tree_random_binary_numeric_split_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/decision_tree/random_binary_numeric_split_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DECISION_TREE_RANDOM_BINARY_NUMERIC_SPLIT_IMPL_HPP
   #define MLPACK_METHODS_DECISION_TREE_RANDOM_BINARY_NUMERIC_SPLIT_IMPL_HPP
   
   #include <mlpack/core/math/random.hpp>
   
   namespace mlpack {
   namespace tree {
   
   template<typename FitnessFunction>
   template<bool UseWeights, typename VecType, typename WeightVecType>
   double RandomBinaryNumericSplit<FitnessFunction>::SplitIfBetter(
       const double bestGain,
       const VecType& data,
       const arma::Row<size_t>& labels,
       const size_t numClasses,
       const WeightVecType& weights,
       const size_t minimumLeafSize,
       const double minimumGainSplit,
       arma::vec& classProbabilities,
       AuxiliarySplitInfo& /* aux */,
       const bool splitIfBetterGain)
   {
     double bestFoundGain = std::min(bestGain + minimumGainSplit, 0.0);
     // Forcing a minimum leaf size of 1 (empty children don't make sense).
     const size_t minimum = std::max(minimumLeafSize, (size_t) 1);
   
     // First sanity check: if we don't have enough points, we can't split.
     if (data.n_elem < (minimum * 2))
       return DBL_MAX;
     if (bestGain == 0.0)
       return DBL_MAX; // It can't be outperformed.
   
     typename VecType::elem_type maxValue = arma::max(data);
     typename VecType::elem_type minValue = arma::min(data);
   
     // Sanity check: if the maximum element is the same as the minimum, we
     // can't split in this dimension.
     if (maxValue == minValue)
       return DBL_MAX;
   
     // Picking a random pivot to split the dimension.
     double randomPivot = math::Random(minValue, maxValue);
   
     // We need to count the number of points for each class.
     arma::Mat<size_t> classCounts;
     arma::mat classWeightSums;
     double totalWeight = 0.0;
     double totalLeftWeight = 0.0;
     double totalRightWeight = 0.0;
     size_t leftLeafSize = 0;
     size_t rightLeafSize = 0;
     if (UseWeights)
     {
       classWeightSums.zeros(numClasses, 2);
       totalWeight = arma::accu(weights);
       bestFoundGain *= totalWeight;
   
       for (size_t i = 0; i < data.n_elem; ++i)
       {
         if (data(i) < randomPivot)
         {
           ++leftLeafSize;
           classWeightSums(labels(i), 0) += weights(i);
           totalLeftWeight += weights(i);
         }
         else
         {
           ++rightLeafSize;
           classWeightSums(labels(i), 1) += weights(i);
           totalRightWeight += weights(i);
         }
       }
     }
     else
     {
       classCounts.zeros(numClasses, 2);
       bestFoundGain *= data.n_elem;
   
       for (size_t i = 0; i < data.n_elem; ++i)
       {
         if (data(i) < randomPivot)
         {
           ++leftLeafSize;
           ++classCounts(labels(i), 0);
         }
         else
         {
           ++rightLeafSize;
           ++classCounts(labels(i), 1);
         }
       }
     }
   
     // Calculate the gain for the left and right child.  Only use weights if
     // needed.
     const double leftGain = UseWeights ?
         FitnessFunction::template EvaluatePtr<true>(classWeightSums.colptr(0),
             numClasses, totalLeftWeight) :
         FitnessFunction::template EvaluatePtr<false>(classCounts.colptr(0),
             numClasses, leftLeafSize);
     const double rightGain = UseWeights ?
         FitnessFunction::template EvaluatePtr<true>(classWeightSums.colptr(1),
             numClasses, totalRightWeight) :
         FitnessFunction::template EvaluatePtr<false>(classCounts.colptr(1),
             numClasses, rightLeafSize);
   
     double gain;
     if (UseWeights)
       gain = totalLeftWeight * leftGain + totalRightWeight * rightGain;
     else
       // Calculate the gain at this split point.
       gain = double(leftLeafSize) * leftGain + double(rightLeafSize) * rightGain;
   
     if (gain < bestFoundGain && splitIfBetterGain)
       return DBL_MAX;
   
     classProbabilities.set_size(1);
     classProbabilities(0) = randomPivot;
   
     if (UseWeights)
       gain /= totalWeight;
     else
       gain /= labels.n_elem;
   
     return gain;
   }
   
   template<typename FitnessFunction>
   template<typename ElemType>
   size_t RandomBinaryNumericSplit<FitnessFunction>::CalculateDirection(
       const ElemType& point,
       const arma::vec& classProbabilities,
       const AuxiliarySplitInfo& /* aux */)
   {
     if (point <= classProbabilities(0))
       return 0; // Go left.
     else
       return 1; // Go right.
   }
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
