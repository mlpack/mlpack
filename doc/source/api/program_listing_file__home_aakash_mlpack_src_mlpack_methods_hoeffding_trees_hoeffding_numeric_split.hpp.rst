
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_hoeffding_numeric_split.hpp:

Program Listing for File hoeffding_numeric_split.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_hoeffding_numeric_split.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/hoeffding_numeric_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP
   #define MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "numeric_split_info.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename FitnessFunction,
            typename ObservationType = double>
   class HoeffdingNumericSplit
   {
    public:
     typedef NumericSplitInfo<ObservationType> SplitInfo;
   
     HoeffdingNumericSplit(const size_t numClasses = 0,
                           const size_t bins = 10,
                           const size_t observationsBeforeBinning = 100);
   
     HoeffdingNumericSplit(const size_t numClasses,
                           const HoeffdingNumericSplit& other);
   
     void Train(ObservationType value, const size_t label);
   
     void EvaluateFitnessFunction(double& bestFitness, double& secondBestFitness)
         const;
   
     size_t NumChildren() const { return bins; }
   
     void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo) const;
   
     size_t MajorityClass() const;
     double MajorityProbability() const;
   
     size_t Bins() const { return bins; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     arma::Col<ObservationType> observations;
     arma::Col<size_t> labels;
   
     arma::Col<ObservationType> splitPoints;
     size_t bins;
     size_t observationsBeforeBinning;
     size_t samplesSeen;
   
     arma::Mat<size_t> sufficientStatistics;
   };
   
   template<typename FitnessFunction>
   using HoeffdingDoubleNumericSplit = HoeffdingNumericSplit<FitnessFunction,
       double>;
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "hoeffding_numeric_split_impl.hpp"
   
   #endif
