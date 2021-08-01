
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_binary_numeric_split.hpp:

Program Listing for File binary_numeric_split.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_binary_numeric_split.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/binary_numeric_split.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_SPLIT_BINARY_NUMERIC_SPLIT_HPP
   #define MLPACK_METHODS_HOEFFDING_SPLIT_BINARY_NUMERIC_SPLIT_HPP
   
   #include "binary_numeric_split_info.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename FitnessFunction,
            typename ObservationType = double>
   class BinaryNumericSplit
   {
    public:
     typedef BinaryNumericSplitInfo<ObservationType> SplitInfo;
   
     BinaryNumericSplit(const size_t numClasses = 0);
   
     BinaryNumericSplit(const size_t numClasses, const BinaryNumericSplit& other);
   
     void Train(ObservationType value, const size_t label);
   
     void EvaluateFitnessFunction(double& bestFitness,
                                  double& secondBestFitness);
   
     // Return the number of children if this node were to split on this feature.
     size_t NumChildren() const { return 2; }
   
     void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo);
   
     size_t MajorityClass() const;
     double MajorityProbability() const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     std::multimap<ObservationType, size_t> sortedElements;
     arma::Col<size_t> classCounts;
   
     ObservationType bestSplit;
     bool isAccurate;
   };
   
   // Convenience typedef.
   template<typename FitnessFunction>
   using BinaryDoubleNumericSplit = BinaryNumericSplit<FitnessFunction, double>;
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "binary_numeric_split_impl.hpp"
   
   #endif
