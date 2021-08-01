
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_hoeffding_categorical_split.hpp:

Program Listing for File hoeffding_categorical_split.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_hoeffding_categorical_split.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/hoeffding_categorical_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_HPP
   #define MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "categorical_split_info.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename FitnessFunction>
   class HoeffdingCategoricalSplit
   {
    public:
     typedef CategoricalSplitInfo SplitInfo;
   
     HoeffdingCategoricalSplit(const size_t numCategories = 0,
                               const size_t numClasses = 0);
   
     HoeffdingCategoricalSplit(const size_t numCategories,
                               const size_t numClasses,
                               const HoeffdingCategoricalSplit& other);
   
     template<typename eT>
     void Train(eT value, const size_t label);
   
     void EvaluateFitnessFunction(double& bestFitness, double& secondBestFitness)
         const;
   
     size_t NumChildren() const { return sufficientStatistics.n_cols; }
   
     void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo);
   
     size_t MajorityClass() const;
     double MajorityProbability() const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(sufficientStatistics));
     }
   
    private:
     arma::Mat<size_t> sufficientStatistics;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "hoeffding_categorical_split_impl.hpp"
   
   #endif
