
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_random_forest_random_forest.hpp:

Program Listing for File random_forest.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_random_forest_random_forest.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/random_forest/random_forest.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_HPP
   #define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_HPP
   
   #include <mlpack/methods/decision_tree/decision_tree.hpp>
   #include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
   #include "bootstrap.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename FitnessFunction = GiniGain,
            typename DimensionSelectionType = MultipleRandomDimensionSelect,
            template<typename> class NumericSplitType = BestBinaryNumericSplit,
            template<typename> class CategoricalSplitType = AllCategoricalSplit,
            bool UseBootstrap = true>
   class RandomForest
   {
    public:
     typedef DecisionTree<FitnessFunction, NumericSplitType, CategoricalSplitType,
         DimensionSelectionType> DecisionTreeType;
   
     RandomForest();
   
     template<typename MatType>
     RandomForest(const MatType& dataset,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const size_t numTrees = 20,
                  const size_t minimumLeafSize = 1,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType>
     RandomForest(const MatType& dataset,
                  const data::DatasetInfo& datasetInfo,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const size_t numTrees = 20,
                  const size_t minimumLeafSize = 1,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType>
     RandomForest(const MatType& dataset,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const arma::rowvec& weights,
                  const size_t numTrees = 20,
                  const size_t minimumLeafSize = 1,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType>
     RandomForest(const MatType& dataset,
                  const data::DatasetInfo& datasetInfo,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const arma::rowvec& weights,
                  const size_t numTrees = 20,
                  const size_t minimumLeafSize = 1,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType>
     double Train(const MatType& data,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const size_t numTrees = 20,
                  const size_t minimumLeafSize = 1,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  const bool warmStart = false,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType>
     double Train(const MatType& data,
                  const data::DatasetInfo& datasetInfo,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const size_t numTrees = 20,
                  const size_t minimumLeafSize = 1,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  const bool warmStart = false,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType>
     double Train(const MatType& data,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const arma::rowvec& weights,
                  const size_t numTrees = 20,
                  const size_t minimumLeafSize = 1,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  const bool warmStart = false,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename MatType>
     double Train(const MatType& data,
                  const data::DatasetInfo& datasetInfo,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const arma::rowvec& weights,
                  const size_t numTrees = 20,
                  const size_t minimumLeafSize = 1,
                  const double minimumGainSplit = 1e-7,
                  const size_t maximumDepth = 0,
                  const bool warmStart = false,
                  DimensionSelectionType dimensionSelector =
                      DimensionSelectionType());
   
     template<typename VecType>
     size_t Classify(const VecType& point) const;
   
     template<typename VecType>
     void Classify(const VecType& point,
                   size_t& prediction,
                   arma::vec& probabilities) const;
   
     template<typename MatType>
     void Classify(const MatType& data,
                   arma::Row<size_t>& predictions) const;
   
     template<typename MatType>
     void Classify(const MatType& data,
                   arma::Row<size_t>& predictions,
                   arma::mat& probabilities) const;
   
     const DecisionTreeType& Tree(const size_t i) const { return trees[i]; }
     DecisionTreeType& Tree(const size_t i) { return trees[i]; }
   
     size_t NumTrees() const { return trees.size(); }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     template<bool UseWeights, bool UseDatasetInfo, typename MatType>
     double Train(const MatType& data,
                  const data::DatasetInfo& datasetInfo,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const arma::rowvec& weights,
                  const size_t numTrees,
                  const size_t minimumLeafSize,
                  const double minimumGainSplit,
                  const size_t maximumDepth,
                  DimensionSelectionType& dimensionSelector,
                  const bool warmStart = false);
   
     std::vector<DecisionTreeType> trees;
   
     double avgGain;
   };
   
   template<typename FitnessFunction = GiniGain,
            typename DimensionSelectionType = MultipleRandomDimensionSelect,
            template<typename> class CategoricalSplitType = AllCategoricalSplit>
   using ExtraTrees = RandomForest<FitnessFunction,
                                   DimensionSelectionType,
                                   RandomBinaryNumericSplit,
                                   CategoricalSplitType,
                                   false>;
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "random_forest_impl.hpp"
   
   #endif
