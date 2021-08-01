
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_hoeffding_tree.hpp:

Program Listing for File hoeffding_tree.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_hoeffding_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/hoeffding_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_HPP
   #define MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/data/dataset_mapper.hpp>
   #include "gini_impurity.hpp"
   #include "hoeffding_numeric_split.hpp"
   #include "hoeffding_categorical_split.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename FitnessFunction = GiniImpurity,
            template<typename> class NumericSplitType =
                HoeffdingDoubleNumericSplit,
            template<typename> class CategoricalSplitType =
                HoeffdingCategoricalSplit
   >
   class HoeffdingTree
   {
    public:
     typedef NumericSplitType<FitnessFunction> NumericSplit;
     typedef CategoricalSplitType<FitnessFunction> CategoricalSplit;
   
     template<typename MatType>
     HoeffdingTree(const MatType& data,
                   const data::DatasetInfo& datasetInfo,
                   const arma::Row<size_t>& labels,
                   const size_t numClasses,
                   const bool batchTraining = true,
                   const double successProbability = 0.95,
                   const size_t maxSamples = 0,
                   const size_t checkInterval = 100,
                   const size_t minSamples = 100,
                   const CategoricalSplitType<FitnessFunction>& categoricalSplitIn
                       = CategoricalSplitType<FitnessFunction>(0, 0),
                   const NumericSplitType<FitnessFunction>& numericSplitIn =
                       NumericSplitType<FitnessFunction>(0));
   
     HoeffdingTree(const data::DatasetInfo& datasetInfo,
                   const size_t numClasses,
                   const double successProbability = 0.95,
                   const size_t maxSamples = 0,
                   const size_t checkInterval = 100,
                   const size_t minSamples = 100,
                   const CategoricalSplitType<FitnessFunction>& categoricalSplitIn
                       = CategoricalSplitType<FitnessFunction>(0, 0),
                   const NumericSplitType<FitnessFunction>& numericSplitIn =
                       NumericSplitType<FitnessFunction>(0),
                   std::unordered_map<size_t, std::pair<size_t, size_t>>*
                       dimensionMappings = NULL,
                   const bool copyDatasetInfo = true);
   
     HoeffdingTree();
   
     HoeffdingTree(const HoeffdingTree& other);
   
     HoeffdingTree(HoeffdingTree&& other);
   
     HoeffdingTree& operator=(const HoeffdingTree& other);
   
     HoeffdingTree& operator=(HoeffdingTree&& other);
   
     ~HoeffdingTree();
   
     template<typename MatType>
     void Train(const MatType& data,
                const arma::Row<size_t>& labels,
                const bool batchTraining = true,
                const bool resetTree = false,
                const size_t numClasses = 0);
   
     template<typename MatType>
     void Train(const MatType& data,
                const data::DatasetInfo& info,
                const arma::Row<size_t>& labels,
                const bool batchTraining = true,
                const size_t numClasses = 0);
   
     template<typename VecType>
     void Train(const VecType& point, const size_t label);
   
     size_t SplitCheck();
   
     size_t SplitDimension() const { return splitDimension; }
   
     size_t MajorityClass() const { return majorityClass; }
     size_t& MajorityClass() { return majorityClass; }
   
     double MajorityProbability() const { return majorityProbability; }
     double& MajorityProbability() { return majorityProbability; }
   
     size_t NumChildren() const { return children.size(); }
   
     const HoeffdingTree& Child(const size_t i) const { return *children[i]; }
     HoeffdingTree& Child(const size_t i) { return *children[i]; }
   
     double SuccessProbability() const { return successProbability; }
     void SuccessProbability(const double successProbability);
   
     size_t MinSamples() const { return minSamples; }
     void MinSamples(const size_t minSamples);
   
     size_t MaxSamples() const { return maxSamples; }
     void MaxSamples(const size_t maxSamples);
   
     size_t CheckInterval() const { return checkInterval; }
     void CheckInterval(const size_t checkInterval);
   
     template<typename VecType>
     size_t CalculateDirection(const VecType& point) const;
   
     template<typename VecType>
     size_t Classify(const VecType& point) const;
   
     size_t NumDescendants() const;
   
     template<typename VecType>
     void Classify(const VecType& point, size_t& prediction, double& probability)
         const;
   
     template<typename MatType>
     void Classify(const MatType& data, arma::Row<size_t>& predictions) const;
   
     template<typename MatType>
     void Classify(const MatType& data,
                   arma::Row<size_t>& predictions,
                   arma::rowvec& probabilities) const;
   
     void CreateChildren();
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     // We need to keep some information for before we have split.
   
     std::vector<NumericSplitType<FitnessFunction>> numericSplits;
     std::vector<CategoricalSplitType<FitnessFunction>> categoricalSplits;
   
     std::unordered_map<size_t, std::pair<size_t, size_t>>* dimensionMappings;
     bool ownsMappings;
   
     size_t numSamples;
     size_t numClasses;
     size_t maxSamples;
     size_t checkInterval;
     size_t minSamples;
     const data::DatasetInfo* datasetInfo;
     bool ownsInfo;
     double successProbability;
   
     // And we need to keep some information for after we have split.
   
     size_t splitDimension;
     size_t majorityClass;
     double majorityProbability;
     typename CategoricalSplitType<FitnessFunction>::SplitInfo categoricalSplit;
     typename NumericSplitType<FitnessFunction>::SplitInfo numericSplit;
     std::vector<HoeffdingTree*> children;
   
     template<typename MatType>
     void TrainInternal(const MatType& data,
                        const arma::Row<size_t>& labels,
                        const bool batchTraining);
   
     void ResetTree(
         const CategoricalSplitType<FitnessFunction>& categoricalSplitIn =
             CategoricalSplitType<FitnessFunction>(0, 0),
         const NumericSplitType<FitnessFunction>& numericSplitIn =
             NumericSplitType<FitnessFunction>(0));
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #include "hoeffding_tree_impl.hpp"
   
   #endif
